"""## Executive summary (read this first)

Call the injected House chat endpoint with strict numeric JSON, explicit thinking
off, a 3,000-token output cap and at most 25 attempts. Never retry or refund an
uncertain request. Stop on HTTP/transport failure and let the forecast retain its
numerical fallback. Do not log credentials, proxy URLs or response error bodies.
"""
import json
import os
import time
import urllib.error
import urllib.request


class HouseUnavailable(RuntimeError):
    pass


class HouseScenarioClient:
    def __init__(self, response_format, *, timeout=180.):
        endpoint = os.environ.get('MODEL_ENDPOINT', '').strip().rstrip('/')
        self.model = os.environ.get('MODEL_NAME', '').strip()
        self._token = os.environ.get('MODEL_TOKEN', '').strip()
        if not endpoint or not self.model or not self._token:
            raise HouseUnavailable('Missing injected House endpoint, model or token')
        self._url = endpoint + ('/chat/completions' if endpoint.endswith('/v1') else '/v1/chat/completions')
        self.response_format, self.timeout = response_format, timeout
        self.calls, self.halted, self.ledger = 0, False, []

    def __call__(self, role, system, prompt, temperature, seed):
        if self.halted:
            raise HouseUnavailable('A prior request failed; no further calls are made')
        if self.calls >= 25:
            raise RuntimeError('House request cap reached')
        if temperature != 1. or type(seed) is not int or not 0 <= seed < 25:
            raise ValueError('Frozen scenario protocol requires temperature 1 and seeds 0..24')
        body = {'model': self.model, 'messages': [{'role': 'system', 'content': system},
                                                 {'role': 'user', 'content': prompt}],
                'temperature': temperature, 'seed': seed, 'max_tokens': 3000,
                'chat_template_kwargs': {'enable_thinking': False},
                'response_format': self.response_format}
        request = urllib.request.Request(self._url, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._token})
        self.calls += 1
        entry = {'role': role, 'seed': seed, 'attempt': self.calls, 'max_tokens': 3000}
        self.ledger.append(entry)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                result = json.load(reply)
            choice = result['choices'][0]
            entry.update(usage=result.get('usage', {}), finish_reason=choice.get('finish_reason'))
            if choice.get('finish_reason') != 'stop':
                entry['invalid'] = 'incomplete reply'
                return None
            try:
                return json.loads(choice.get('message', {}).get('content') or '')
            except (ValueError, TypeError):
                entry['invalid'] = 'non-JSON reply'
                return None
        except Exception as exc:
            self.halted = True
            # An upstream error may echo headers. Record only its type/status.
            entry['service_error'] = ('HTTP ' + str(exc.code) if isinstance(exc, urllib.error.HTTPError)
                                      else type(exc).__name__)
            raise HouseUnavailable(entry['service_error']) from None
        finally:
            entry['seconds'] = time.monotonic() - started
