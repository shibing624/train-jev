"""System One transport, image placement and pixel-to-readout tests without downloading a model."""

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import trainjev.assets as assets_module
import trainjev.serve as srv
from trainjev.assets import MARKER_PREFIX, VISION_TOKEN


class Inputs(dict):
    def to(self, device):
        return Inputs({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    choice_labels = list('ABCDEFGHIJKLMNOP') + [f'L{i}' for i in range(16, 256)]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[1] * 12)


class Processor:
    def __init__(self):
        self.calls = []
        self.tokens = 18

    def __call__(self, *, text, images, return_tensors, truncation):
        assert return_tensors == 'pt' and truncation is False
        assert text[0].count(VISION_TOKEN) == len(images)
        assert MARKER_PREFIX not in text[0]
        colors = [image.getpixel((0, 0)) for image in images]
        self.calls.append((text[0], colors))
        value = sum((color[0] - color[2]) / 255 for color in colors)
        return Inputs(input_ids=torch.ones((1, self.tokens), dtype=torch.long),
                      pixel_values=torch.tensor([value]))


class Vision:
    def __init__(self):
        self.calls = 0

    def __call__(self, **inputs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=inputs['pixel_values'].reshape(1, 1, 1))


class TextBody:
    def __call__(self, input_ids):
        return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 1)))


class Head:
    attention_mode = "causal"

    def logits(self, h, kind, n):
        if kind == 'noul':
            return torch.stack([-4 * h[0], 4 * h[0]])
        return torch.arange(n, dtype=torch.float32) * h[0]


@pytest.fixture
def client():
    srv.S.clear()
    srv.S.update(args=srv.parse_args(['--vision', '--device', 'cpu']), tok=Tokenizer(), body=TextBody(),
                 head=Head(), temps={'choice': 1, 'noul': 1, 'score': 1},
                 processor=Processor(), vision=Vision(), lock=asyncio.Lock())
    with TestClient(srv.build_app()) as test_client:
        yield test_client
    srv.S.clear()


def raw_image(color='red'):
    output = io.BytesIO()
    Image.new('RGB', (280, 280), color).save(output, format='PNG')
    return output.getvalue()


def asset(color='red'):
    return {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(raw_image(color)).decode()}


def body(color='red'):
    return {'model': 'jev-latest', 'state': {'door': 'uuid-photo-123'},
            'assets': {'uuid-photo-123': asset(color)},
            'questions': {'person_present': {'type': 'noul', 'instructions': 'Is the image in door red?'}}}


def test_base64_pixels_reach_the_model_and_change_decision(client):
    red = client.post('/v1/systemone', json=body('red'))
    blue = client.post('/v1/systemone', json=body('blue'))
    assert red.status_code == blue.status_code == 200
    assert red.json()['answers']['person_present']['noul'] > .99
    assert blue.json()['answers']['person_present']['noul'] < .01
    assert red.json()['usage']['input_tokens'] == 18
    assert srv.S['vision'].calls == 2
    assert client.get('/health').json()['images'] is True


def test_multipart_and_json_are_equivalent(client):
    expected = client.post('/v1/systemone', json=body()).json()['answers']
    request = body()
    request['assets'] = {'uuid-photo-123': {'type': 'image', 'part': 'door-photo'}}
    response = client.post('/v1/systemone', files=[
        ('request', (None, json.dumps(request), 'application/json')),
        ('door-photo', ('door.png', raw_image(), 'image/png')),
    ])
    assert response.status_code == 200, response.text
    assert response.json()['answers'] == expected


def test_question_only_images_do_not_leak_into_other_questions(client):
    request = body()
    request['state'] = 'shared text only'
    request['questions'] = {
        'image_question': {'type': 'noul', 'instructions': 'Is uuid-photo-123 red?'},
        'text_question': {'type': 'noul', 'instructions': 'Does shared text mention red?'},
    }
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 200
    assert response.json()['answers']['image_question']['noul'] > .99
    assert response.json()['answers']['text_question']['noul'] == .5
    assert srv.S['vision'].calls == 1
    assert len(srv.S['processor'].calls) == 1


def test_state_and_nested_criteria_images_preserve_option_keys(client):
    request = body()
    request['assets']['uuid-example-456'] = asset('blue')
    request['questions'] = {
        'carrying': {'type': 'choice', 'instructions': 'Compare door with the examples.',
                     'criteria': {'package': {'examples': ['uuid-example-456']}, 'nothing': None}},
        'presence': {'type': 'noul', 'instructions': 'Is door red?'},
    }
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 200
    assert set(response.json()['answers']['carrying']['probabilities']) == {'package', 'nothing'}
    assert [len(colors) for _, colors in srv.S['processor'].calls] == [2, 1]
    assert MARKER_PREFIX not in json.dumps(response.json())


@pytest.mark.parametrize('levels', [3, 6])
def test_image_score_maps_to_expected_level_distribution(client, levels):
    request = body()
    request['questions'] = {'rating': {'type': 'score', 'instructions': 'How red is door?',
                                     'criteria': [f'level {i}' for i in range(levels)]}}
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 200
    answer = response.json()['answers']['rating']
    assert set(answer['probabilities']) == {str(i) for i in range(levels)}
    assert sum(answer['probabilities'].values()) == pytest.approx(1)
    assert 0 <= answer['score'] <= levels - 1


@pytest.mark.parametrize('strategy', ['single', 'tournament', 'permute'])
def test_wide_image_choices_use_the_existing_strategies(client, strategy):
    request = body()
    request['strategy'] = strategy
    request['questions'] = {'q': {'type': 'choice', 'instructions': 'Pick a level.',
                                 'criteria': {f'option{i}': None for i in range(20)}}}
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 200, response.text
    probabilities = response.json()['answers']['q']['probabilities']
    assert len(probabilities) == 20 and sum(probabilities.values()) == pytest.approx(1)
    assert srv.S['vision'].calls == {'single': 1, 'tournament': 3, 'permute': 4}[strategy]


def test_image_context_is_rejected_without_inference_or_truncation(client):
    srv.S['args'].max_len = 10
    response = client.post('/v1/systemone', json=body())
    assert response.status_code == 422 and 'never silently truncated' in response.json()['error']
    assert srv.S['vision'].calls == 0


def test_oversized_repeated_images_rejected_before_patch_allocation(client):
    srv.S['processor'].image_processor = SimpleNamespace(patch_size=16, merge_size=2,
                                                         size={'longest_edge': 16777216})
    output = io.BytesIO()
    Image.new('RGB', (1568, 1568), 'red').save(output, format='PNG')
    request = body()
    request['state'] = ['uuid-photo-123'] * 15
    request['assets']['uuid-photo-123']['data_base64'] = base64.b64encode(output.getvalue()).decode()
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 422
    assert not srv.S['processor'].calls and srv.S['vision'].calls == 0


def test_memory_exhaustion_returns_retryable_error_without_partial_answers(client, monkeypatch):
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError('private backend detail')

    monkeypatch.setattr(srv, '_hf_image', oom)
    response = client.post('/v1/systemone', json=body())
    assert response.status_code == 503 and 'answers' not in response.json()
    assert 'private' not in response.json()['error']


def test_text_only_backend_rejects_images_and_retains_text(client, monkeypatch):
    srv.S['vision'] = None
    srv.S['args'].vision = False
    monkeypatch.setattr(assets_module, '_prepare_image',
                        lambda *a, **k: pytest.fail('text-only serving must not decode image assets'))
    response = client.post('/v1/systemone', json=body())
    assert response.status_code == 422 and '--vision' in response.json()['error']
    request = {'state': 'plain state', 'questions': {'q': {'type': 'noul', 'instructions': 'plain question'}}}
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 200 and response.json()['answers']['q']['noul'] == .5
    assert client.get('/health').json()['images'] is False


def test_bad_images_fail_whole_request_before_model(client):
    request = body()
    request['assets']['uuid-bad-456'] = {'format': 'png', 'data_base64': 'not base64'}
    request['questions']['other'] = {'type': 'noul', 'instructions': 'Is uuid-bad-456 blue?'}
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 400 and 'answers' not in response.json()
    assert srv.S['vision'].calls == 0
    request['assets']['uuid-bad-456'] = {'url': 'http://127.0.0.1/private'}
    assert client.post('/v1/systemone', json=request).status_code == 400


@pytest.mark.parametrize('payload', [{}, {'state': 'x', 'questions': {}}, {'state': 'x', 'questions': []},
                                     {'state': 'x', 'questions': {'q': {'type': 'box'}}},
                                     {'state': 'x', 'questions': {'q': {'type': 'choice', 'criteria': []}}}])
def test_invalid_questions_return_422(client, payload):
    assert client.post('/v1/systemone', json=payload).status_code == 422


def test_actual_endpoint_applies_body_limit(client, monkeypatch):
    monkeypatch.setattr(assets_module, 'MAX_BODY_BYTES', 8)
    response = client.post('/v1/systemone', content=b'{"state": "too big"}',
                           headers={'content-type': 'application/json'})
    assert response.status_code == 413


def test_proxy_keeps_all_question_and_criterion_images(client):
    calls = []

    class Proxy:
        async def post(self, path, *, json):
            calls.append(json)
            assert path == '/v1/decide' and json['thinking'] == 'off'
            n = len(json['options']) if json['kind'] == 'choice' else 2
            return SimpleNamespace(status_code=200, json=lambda: {'probabilities': [1/n] * n,
                                                                  'usage': {'prompt_tokens': 35}})

    srv.S['args'].proxy = 'http://configured-upstream'
    srv.S['client'] = Proxy()
    request = body()
    request['assets']['uuid-example-456'] = asset('blue')
    request['questions'] = {
        'q': {'type': 'choice', 'instructions': 'Compare door to uuid-example-456.',
              'criteria': {'package': 'uuid-example-456', 'nothing': None}},
        'plain': {'type': 'noul', 'instructions': 'Is door red?'},
    }
    response = client.post('/v1/systemone', json=request)
    assert response.status_code == 200
    assert [sum(isinstance(part, dict) for part in call['state']) for call in calls] == [3, 1]
    assert all(call['strategy'] == 'single' for call in calls)
    assert all(MARKER_PREFIX not in json.dumps(call) for call in calls)
    assert 'question image' in calls[0]['question'] and 'option 0 image' in calls[0]['options'][0]
    assert response.json()['usage']['input_tokens'] == 70


def test_setup_vision_uses_full_loader_and_shares_text_body(monkeypatch):
    srv.S.clear()
    text, processor, vision = object(), object(), object()
    head = Head()
    monkeypatch.setattr(srv, 'load_vision_model', lambda *a, **k: (Tokenizer(), text, head, processor, vision))
    monkeypatch.setattr(srv, 'load_model', lambda *a, **k: pytest.fail('vision setup used text-only loader'))
    srv.setup(srv.parse_args(['--vision', '--base_only', '--device', 'cpu']))
    assert srv.S['body'] is text and srv.S['vision'] is vision and srv.S['processor'] is processor
    srv.S.clear()


@pytest.mark.parametrize('transport', ['multipart', 'base64'])
def test_image_example_uses_compatible_request(client, tmp_path, monkeypatch, transport):
    from examples.image_decisions import main

    path = tmp_path / 'door.png'
    path.write_bytes(raw_image())

    def send(url, **kwargs):
        assert url == 'http://127.0.0.1:8020/v1/systemone'
        assert 'headers' not in kwargs
        kwargs.pop('timeout')
        return client.post('/v1/systemone', **kwargs)

    monkeypatch.setattr('examples.image_decisions.httpx.post', send)
    result = main(['--image', str(path), '--transport', transport])
    assert result['person_present']['type'] == 'noul'
    assert set(result['carrying']['probabilities']) == {'package', 'bag', 'nothing'}


@pytest.mark.parametrize('value', ['true', 'false'])
def test_legacy_image_header_is_ignored(client, value):
    expected = client.post('/v1/systemone', json=body()).json()['answers']
    response = client.post('/v1/systemone', json=body(), headers={'TypeSafe-Danger-Zone-Images': value})
    assert response.status_code == 200
    assert response.json()['answers'] == expected
