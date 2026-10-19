"""Fixed-input Bamboo/StartLux API evaluation and a local System One proxy for memwrite8.

Credentials come only from BAMBOO_API_KEY. Never use the evaluation rows for training or tuning.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from scripts.evaluate_pk import prediction, summarise
from scripts.prepare_eval_data import load_records

DEFAULT_ENDPOINT = 'https://api.bamboonode.dev/v1/decisions'
DEFAULT_MODEL = 'startlux-decision-27b-autothink'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def row_questions(row):
    q = {'type': row['kind'], 'instructions': row['question']}
    if row['kind'] == 'choice':
        # Bare numeric wire IDs are hidden by StartLux's formatter; retain every option verbatim as its criterion.
        q['criteria'] = {str(i): option for i, option in enumerate(row['options'])}
    elif row['kind'] == 'score':
        q['criteria'] = row['options']
    return {'answer': q}


def read_answer(row, result):
    answer = result['answers']['answer']
    if answer['type'] != row['kind']:
        raise ValueError('Answer type changed')
    if row['kind'] == 'noul':
        if row['options'] != ['false', 'true']:
            raise ValueError('Noul option order must be false, true')
        value = answer['noul']
        if type(value) not in (int, float):
            raise ValueError('Noul must be a numeric probability')
        probabilities = [1-value, value]
    else:
        keys = [str(i) for i in range(len(row['options']))]
        if set(answer['probabilities']) != set(keys):
            raise ValueError('Answer option keys changed')
        probabilities = [answer['probabilities'][key] for key in keys]
        if row['kind'] == 'choice' and answer['choice'] != keys[prediction('choice', probabilities)]:
            raise ValueError('Returned choice does not match highest probability')
    if (len(probabilities) != len(row['target'])
            or not all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)):
        raise ValueError('Invalid probability vector')
    return probabilities


class BambooClient:
    def __init__(self, args, *, transport=None):
        self.key = os.environ.get('BAMBOO_API_KEY', '')
        if not self.key:
            raise ValueError('Set BAMBOO_API_KEY; no credential is embedded in this script')
        self.endpoint = args.endpoint
        self.controls = {'model': args.model, 'think_mode': args.mode, 'gate': args.gate,
                         'think_budget': args.think_budget, 'mix': args.mix,
                         'none_check': False, 'allow_none': False,
                         'return_details': True, 'return_thought': False}
        self.mode = args.mode
        self.rps = args.rps
        self.retries = args.retries
        self.clock = asyncio.Lock()
        self.next_start = 0.0
        self.client = httpx.AsyncClient(timeout=args.timeout, transport=transport, headers={
            'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        self.requests = 0
        self.attempts = 0
        self.statuses = Counter()

    def redact(self, text):
        return str(text).replace(self.key, '[REDACTED]')

    async def close(self):
        await self.client.aclose()

    async def pace(self):
        start = time.perf_counter()
        async with self.clock:
            delay = self.next_start - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = time.perf_counter() + 1/self.rps
        return time.perf_counter() - start

    async def evaluate(self, state, questions):
        payload = {'state': state, 'questions': questions, **self.controls}
        attempts = []
        start = time.perf_counter()
        for attempt in range(self.retries + 1):
            wait_seconds = await self.pace()
            sent = time.perf_counter()
            try:
                response = await self.client.post(self.endpoint, json=payload)
                self.attempts += 1
                self.statuses[str(response.status_code)] += 1
                ms = (time.perf_counter()-sent)*1000
                attempts.append({'status': response.status_code, 'latency_ms': ms,
                                 'pacing_wait_ms': wait_seconds*1000})
                if response.status_code in (401, 403):
                    raise PermissionError(f'Bamboo authentication/permission failure: HTTP {response.status_code}')
                if response.status_code == 200:
                    result = json.loads(self.redact(response.text))
                    if result.get('model') != self.controls['model']:
                        raise ValueError('Remote model identity changed')
                    if set(result['answers']) != set(questions):
                        raise ValueError('Missing or extra answer keys')
                    for answer in result['answers'].values():
                        detail = answer.get('details', {})
                        thought = detail.get('thought')
                        if not isinstance(thought, dict):
                            raise ValueError('Missing requested thought usage details')
                        if self.mode == 'never' and (thought.get('used') or thought.get('tokens', 0)):
                            raise ValueError('Server thought despite think_mode=never')
                        if self.mode == 'auto' and detail.get('gate') not in (None, self.controls['gate']):
                            raise ValueError('Server ignored the frozen gate')
                    self.requests += 1
                    return result, {'latency_ms': ms, 'request_total_ms': (time.perf_counter()-start)*1000,
                                    'attempts': attempts}
                if response.status_code not in (429, 500, 502, 503, 504, 529):
                    raise ValueError(self.redact(f'HTTP {response.status_code}: {response.text[:1000]}'))
                try:
                    delay = float(response.headers.get('retry-after', .5 * 2**attempt))
                except ValueError:
                    delay = .5 * 2**attempt
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                self.attempts += 1
                self.statuses[type(error).__name__] += 1
                attempts.append({'status': type(error).__name__, 'latency_ms': (time.perf_counter()-sent)*1000})
                delay = .5 * 2**attempt
            if attempt == self.retries:
                raise RuntimeError('Retry budget exhausted; request counted as failure, not silently omitted')
            await asyncio.sleep(min(max(delay, .1), 30))
        raise RuntimeError('Unreachable request loop')


def protocol(args, rows):
    return {'endpoint': args.endpoint, 'model': args.model, 'mode': args.mode, 'gate': args.gate,
            'think_budget': args.think_budget, 'mix': args.mix, 'none_check': False, 'allow_none': False,
            'dataset_sha256': sha(Path(args.dataset)/'rows.jsonl'), 'rows': len(rows),
            'concurrency': args.concurrency, 'rps': args.rps, 'retries': args.retries,
            'return_details': True, 'return_thought': False,
            'choice_wire_keys': 'numeric indices; complete original option text in criteria; formatter hides IDs',
            'note': 'Original inputs/options/labels fixed; hosted alias is not a pinned weight revision. '
                    'No recalibration, gold-conditioned prompts or evaluation-based gate tuning.'}


async def evaluate_pk(args):
    rows = load_records(args.dataset)
    if len(rows) != 2494 and not args.allow_other_size:
        raise ValueError('Expected the complete 2494-row PK; incomplete/subsampled scores refused')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expected = protocol(args, rows)
    protocol_file = out/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != expected:
            raise ValueError('Resume protocol differs from the existing run')
    else:
        save(protocol_file, expected)
    path = out/'predictions.jsonl'
    latest = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                latest[record['id']] = record
    by_id = {row['id']: row for row in rows}
    if set(latest)-set(by_id):
        raise ValueError('Existing results have foreign row IDs')
    for rid, result in latest.items():
        if result['input_sha256'] != by_id[rid]['input_sha256']:
            raise ValueError('Existing result input differs')
    pending = iter(row for row in rows if latest.get(row['id'], {}).get('status') != 'ok')
    write_lock = asyncio.Lock()
    client = BambooClient(args)
    started = time.perf_counter()
    finished = 0

    async def worker():
        nonlocal finished
        for row in pending:
            record = {k: row[k] for k in ('id', 'task', 'eval_group', 'group_id', 'input_sha256', 'kind', 'target')}
            try:
                response, timing = await client.evaluate(row['state'], row_questions(row))
                probs = read_answer(row, response)
                pred = prediction(row['kind'], probs)
                thought = response['answers']['answer']['details']['thought']
                record.update(status='ok', probs=probs, prediction=pred, correct=pred==row['target'].index(1),
                              response=response, thought_used=bool(thought['used']),
                              thought_tokens=int(thought['tokens']),
                              **timing)
            except PermissionError:
                raise
            except Exception as error:
                record.update(status='error', error=client.redact(f'{type(error).__name__}: {error}'))
            async with write_lock:
                with path.open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                latest[row['id']] = record
                finished += 1
                if finished % 25 == 0:
                    print(json.dumps({'completed_this_phase': finished,
                                      'latest_ok': sum(r['status']=='ok' for r in latest.values()),
                                      'requests': len(rows), 'seconds': round(time.perf_counter()-started, 2),
                                      'http_statuses': dict(client.statuses)}), flush=True)
    try:
        async with asyncio.TaskGroup() as group:
            for _ in range(args.concurrency):
                group.create_task(worker())
    finally:
        await client.close()
        save(out/'run_status.json', {'seconds_this_phase': time.perf_counter()-started,
                                     'http_attempts_this_phase': client.attempts,
                                     'http_statuses_this_phase': dict(client.statuses),
                                     'latest_ok': sum(r['status']=='ok' for r in latest.values()),
                                     'latest_errors': sum(r['status']!='ok' for r in latest.values()),
                                     'expected': len(rows)})
    if len(latest) != len(rows) or any(r['status'] != 'ok' for r in latest.values()):
        raise RuntimeError('PK incomplete or has failed rows; inspect predictions/run_status, no full score emitted')
    probabilities = [latest[row['id']]['probs'] for row in rows]
    result = summarise(rows, probabilities)
    ms = sorted(latest[row['id']]['latency_ms'] for row in rows)
    tokens = [latest[row['id']]['thought_tokens'] for row in rows]
    result['runtime'] = {'latency_p50_ms': ms[len(ms)//2], 'latency_p95_ms': ms[int(.95*(len(ms)-1))],
                         'thought_used': sum(latest[row['id']]['thought_used'] for row in rows),
                         'thought_tokens_total': sum(tokens), 'thought_tokens_mean': sum(tokens)/len(rows),
                         'thought_tokens_max': max(tokens),
                         'latency_note': 'API round trip including network/proxy, not local GPU latency'}
    result['protocol'] = expected
    result['complete'] = True
    save(out/'summary.json', result)
    print(json.dumps({'complete': True, 'macro': result['macro'], 'runtime': result['runtime']}), flush=True)


def serve_proxy(args):
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()
    client = BambooClient(args)

    @app.get('/health')
    async def health():
        return {'ok': True, 'controls': client.controls, 'successful_requests': client.requests,
                'http_attempts': client.attempts, 'http_statuses': dict(client.statuses)}

    @app.post('/v1/systemone')
    async def systemone(request: Request):
        body = await request.json()
        try:
            result, timing = await client.evaluate(body['state'], body['questions'])
            result['api_client_timing'] = timing
            return result
        except Exception as error:
            return JSONResponse({'error': client.redact(f'{type(error).__name__}: {error}')}, status_code=502)

    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--mode', choices=('auto', 'never'), required=True)
    p.add_argument('--gate', type=float, default=.9)
    p.add_argument('--think-budget', dest='think_budget', type=int, default=4096)
    p.add_argument('--mix', type=float, default=.75)
    p.add_argument('--rps', type=float, default=3)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--dataset', default='data/train-jev-pk')
    p.add_argument('--out')
    p.add_argument('--allow-other-size', action='store_true', help='for explicit smoke/unit fixtures only')
    p.add_argument('--serve-proxy', action='store_true')
    p.add_argument('--port', type=int, default=8071)
    args = p.parse_args(argv)
    if not (all(math.isfinite(v) for v in (args.gate, args.mix, args.rps, args.timeout))
            and 0 <= args.gate <= 1 and 0 <= args.mix <= 1 and args.think_budget > 0 and args.rps > 0
            and args.concurrency > 0 and args.timeout > 0 and args.retries >= 0):
        p.error('Invalid budgets, rates or probabilities')
    if not args.serve_proxy and not args.out:
        p.error('--out is required for PK evaluation')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.serve_proxy:
        serve_proxy(arguments)
    else:
        asyncio.run(evaluate_pk(arguments))
