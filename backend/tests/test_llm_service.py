import httpx
import openai
import pytest

from app.core.config import Settings
from app.services import llm_service
from app.services.llm_service import LLMNotConfiguredError, LLMRequestError, LLMService, LLMUnavailableError


def make_service(**kw) -> LLMService:
    return LLMService(Settings(llm_api_key="sk-test", llm_model="m", llm_max_attempts=3, _env_file=None, **kw))


def err(cls, status=429):
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    if cls is openai.APIConnectionError:
        return cls(request=req)
    return cls("boom", response=httpx.Response(status, request=req), body=None)


class Resp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls, self.params = list(outcomes), 0, []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **params):
        self.calls += 1
        self.params.append(params)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return Resp(out)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_service.time, "sleep", lambda s: None)


def test_not_configured_raises_clear_error():
    with pytest.raises(LLMNotConfiguredError):
        LLMService(Settings(_env_file=None)).chat([{"role": "user", "content": "hi"}])


def test_rate_limit_is_retried_then_succeeds():
    svc = make_service()
    svc._client = FakeClient([err(openai.RateLimitError), err(openai.RateLimitError), "ok"])
    assert svc.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert svc._client.calls == 3


def test_persistent_rate_limit_gives_unavailable_after_bounded_retries():
    svc = make_service()
    svc._client = FakeClient([err(openai.RateLimitError)] * 5)
    with pytest.raises(LLMUnavailableError):
        svc.chat([{"role": "user", "content": "hi"}])
    assert svc._client.calls == 3  # never retries forever


def test_auth_error_is_not_retried_and_hides_key():
    svc = make_service()
    svc._client = FakeClient([err(openai.AuthenticationError, 401)] * 3)
    with pytest.raises(LLMRequestError) as e:
        svc.chat([{"role": "user", "content": "hi"}])
    assert svc._client.calls == 1 and "sk-test" not in str(e.value)


def test_connection_error_retried():
    svc = make_service()
    svc._client = FakeClient([err(openai.APIConnectionError), "fine"])
    assert svc.chat([{"role": "user", "content": "x"}]) == "fine"


def test_empty_response_is_an_error():
    svc = make_service(); svc._client = FakeClient([""] * 3)
    with pytest.raises(LLMUnavailableError):
        svc.chat([{"role": "user", "content": "x"}])


def test_provider_parameter_differences_are_configurable():
    svc = make_service(llm_max_tokens_param="max_completion_tokens", llm_send_temperature=False)
    svc._client = FakeClient(["ok"])
    svc.chat([{"role": "user", "content": "x"}])
    p = svc._client.params[0]
    assert "temperature" not in p and "max_completion_tokens" in p and "max_tokens" not in p
