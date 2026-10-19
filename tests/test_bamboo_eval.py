"""CPU-only API evaluation tests with a mock transport; no credentials or network calls."""
import asyncio
import hashlib
import json

import httpx
import pytest

import scripts.evaluate_bamboo as api


def row(kind='choice'):
    options = ['red', 'blue'] if kind == 'choice' else ['false', 'true'] if kind == 'noul' else list('012345')
    return {'id': 'r1', 'kind': kind, 'state': 'A red parcel.', 'question': 'Which color?', 'options': options,
            'target': [1] + [0] * (len(options)-1), 'task': 'test/color', 'eval_group': 'test', 'group_id': 'group1',
            'source': {'file': 'color.jsonl', 'row_index': 0, 'native_label': 'red', 'text_limit': None}}


def response(payload, *, used=False):
    answers = {}
    for qid, q in payload['questions'].items():
        keys = list(q['criteria'])
        p = dict(zip(keys, (.8, .2)))
        answers[qid] = {'type': 'choice', 'choice': keys[0], 'probabilities': p,
                        'details': {'gate': payload['gate'], 'thought': {'used': used, 'tokens': 12 if used else 0}}}
    return {'model': payload['model'], 'answers': answers, 'usage': {'input_tokens': 30, 'output_tokens': 0}}


def args(tmp_path, mode='never'):
    return api.parse_args(['--mode', mode, '--out', str(tmp_path/'out'), '--rps', '1000', '--allow-other-size'])


def test_payload_never_contains_gold_or_source_metadata():
    r = row()
    q = api.row_questions(r)
    assert q == {'answer': {'type': 'choice', 'instructions': r['question'], 'criteria': {'0': 'red', '1': 'blue'}}}
    assert 'target' not in json.dumps(q) and 'source' not in json.dumps(q)


def test_native_types_and_noul_tie():
    assert api.read_answer(row(), {'answers': {'answer': {'type': 'choice', 'choice': '0',
                                                         'probabilities': {'0': .8, '1': .2}}}}) == [.8, .2]
    r = row('noul')
    assert api.read_answer(r, {'answers': {'answer': {'type': 'noul', 'noul': .5}}}) == [.5, .5]
    assert api.prediction('noul', [.5, .5]) == 1
    r = row('score')
    p = {str(i): 1/6 for i in range(6)}
    assert api.read_answer(r, {'answers': {'answer': {'type': 'score', 'probabilities': p}}}) == [1/6]*6


@pytest.mark.parametrize('answer', [
    {'type': 'choice', 'choice': '0', 'probabilities': {'0': .2, '1': .8}},
    {'type': 'choice', 'choice': '0', 'probabilities': {'0': .8, 'extra': .2}},
    {'type': 'choice', 'choice': '0', 'probabilities': {'0': float('nan'), '1': .2}},
])
def test_invalid_answer_refused(answer):
    with pytest.raises(ValueError):
        api.read_answer(row(), {'answers': {'answer': answer}})


def test_retry_and_secret_redaction(monkeypatch, tmp_path):
    monkeypatch.setenv('BAMBOO_API_KEY', 'unit-test-secret')
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert set(payload) == {'state', 'questions', 'model', 'think_mode', 'gate', 'think_budget', 'mix',
                                'none_check', 'allow_none', 'return_details', 'return_thought'}
        assert not payload['none_check'] and not payload['allow_none']
        if len(calls) == 1:
            return httpx.Response(429, headers={'retry-after': '.01'}, json={'error': 'limit'})
        result = response(payload)
        result['echo'] = 'unit-test-secret'
        return httpx.Response(200, json=result)

    async def run():
        client = api.BambooClient(args(tmp_path), transport=httpx.MockTransport(handler))
        result, timing = await client.evaluate(row()['state'], api.row_questions(row()))
        assert result['echo'] == '[REDACTED]' and len(timing['attempts']) == 2
        assert client.statuses == {'429': 1, '200': 1}
        await client.close()

    asyncio.run(run())
    assert len(calls) == 2


def test_auth_failure_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv('BAMBOO_API_KEY', 'unit-test-secret')
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={'error': 'authentication failure'})

    async def run():
        client = api.BambooClient(args(tmp_path), transport=httpx.MockTransport(handler))
        with pytest.raises(PermissionError):
            await client.evaluate(row()['state'], api.row_questions(row()))
        await client.close()

    asyncio.run(run())
    assert len(calls) == 1


def test_never_mode_cannot_silently_think(monkeypatch, tmp_path):
    monkeypatch.setenv('BAMBOO_API_KEY', 'unit-test-secret')

    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response(
            json.loads(request.content), used=True)))
        client = api.BambooClient(args(tmp_path), transport=transport)
        with pytest.raises(ValueError, match='despite'):
            await client.evaluate(row()['state'], api.row_questions(row()))
        await client.close()

    asyncio.run(run())


def test_resume_validates_protocol_and_skips_success(monkeypatch, tmp_path):
    monkeypatch.setenv('BAMBOO_API_KEY', 'unit-test-secret')
    dataset = tmp_path/'data'
    dataset.mkdir()
    r = row()
    raw_input = json.dumps({k: r[k] for k in ('kind', 'state', 'question', 'options')},
                          ensure_ascii=False, separators=(',', ':')).encode()
    r['input_sha256'] = hashlib.sha256(raw_input).hexdigest()
    raw = (json.dumps(r)+'\n').encode()
    (dataset/'rows.jsonl').write_bytes(raw)
    (dataset/'manifest.json').write_text(json.dumps({'rows': 1, 'files': {'rows.jsonl': {
        'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}}}))
    a = args(tmp_path)
    a.dataset = str(dataset)
    calls = []
    original = api.BambooClient

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response(json.loads(request.content)))

    monkeypatch.setattr(api, 'BambooClient', lambda a: original(a, transport=httpx.MockTransport(handler)))
    asyncio.run(api.evaluate_pk(a))
    asyncio.run(api.evaluate_pk(a))
    assert len(calls) == 1
    summary = json.loads((tmp_path/'out/summary.json').read_text())
    assert summary['complete'] and summary['rows'] == 1
    assert summary['runtime']['thought_tokens_total'] == 0
    a.gate = .8
    with pytest.raises(ValueError, match='Resume protocol'):
        asyncio.run(api.evaluate_pk(a))


def test_requires_credentials_and_finite_limits(monkeypatch, tmp_path):
    monkeypatch.delenv('BAMBOO_API_KEY', raising=False)
    with pytest.raises(ValueError, match='BAMBOO_API_KEY'):
        api.BambooClient(args(tmp_path))
    with pytest.raises(SystemExit):
        api.parse_args(['--mode', 'auto', '--out', str(tmp_path), '--rps', 'inf'])
