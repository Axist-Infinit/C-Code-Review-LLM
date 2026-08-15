"""Tests for providers.py — the backend seam (no network, no langchain needed).

The point of this module is that the reviewer's logic never learns which model
answered, so these tests pin the three things a caller does depend on: how a
spec resolves, which of the three failure classes a given breakage produces,
and that a reply survives the round trip whatever shape the provider returns.
"""
import json
import urllib.error

import pytest

import providers as p


# --- spec parsing ------------------------------------------------------------
# A bare Ollama tag and a provider-qualified spec both contain a colon; the
# split rule is what keeps existing profiles.json / command lines working.

@pytest.mark.parametrize("spec,expect", [
    ("qwen2.5-coder:14b", ("ollama", "qwen2.5-coder:14b")),
    ("qwen2.5-coder", ("ollama", "qwen2.5-coder")),
    ("ollama:qwen2.5-coder:32b", ("ollama", "qwen2.5-coder:32b")),
    ("anthropic:claude-opus-5", ("anthropic", "claude-opus-5")),
    ("openai:gpt-5", ("openai", "gpt-5")),
    ("lc-ollama:qwen2.5-coder:7b", ("lc-ollama", "qwen2.5-coder:7b")),
    ("  anthropic:claude-opus-5  ", ("anthropic", "claude-opus-5")),
    ("ANTHROPIC:claude-opus-5", ("anthropic", "claude-opus-5")),
])
def test_parse_spec(spec, expect):
    assert p.parse_spec(spec) == expect


def test_unknown_prefix_stays_an_ollama_tag():
    # "deepseek-coder-v2:16b" is an Ollama TAG whose first token merely looks
    # like a provider name. Splitting it would ask deepseek's API for "16b".
    assert p.parse_spec("deepseek-coder-v2:16b") == ("ollama", "deepseek-coder-v2:16b")


def test_empty_spec_rejected():
    with pytest.raises(ValueError):
        p.parse_spec("  ")


@pytest.mark.parametrize("spec,local", [
    ("qwen2.5-coder:14b", True),
    ("ollama:qwen2.5-coder:14b", True),
    ("lc-ollama:qwen2.5-coder:7b", True),
    ("anthropic:claude-opus-5", False),
    ("openai:gpt-5", False),
])
def test_is_local(spec, local):
    assert p.is_local(spec) is local


def test_egress_warning_only_fires_for_remote_specs():
    assert p.egress_warning(["ollama:a", "lc-ollama:b"]) == ""
    warn = p.egress_warning(["ollama:a", "openai:gpt-5"])
    assert "openai:gpt-5" in warn and "ollama:a" not in warn


# --- spec resolution precedence ----------------------------------------------

def test_resolve_specs_precedence(monkeypatch):
    prof = {"model": "anthropic:claude-opus-5", "ollama_model": "qwen2.5-coder:14b"}
    monkeypatch.delenv("CCR_MODEL", raising=False)
    assert p.resolve_specs("openai:gpt-5", profile=prof) == ["openai:gpt-5"]
    monkeypatch.setenv("CCR_MODEL", "groq:llama-3.3")
    assert p.resolve_specs(None, profile=prof) == ["groq:llama-3.3"]
    monkeypatch.delenv("CCR_MODEL")
    assert p.resolve_specs(None, profile=prof) == ["anthropic:claude-opus-5"]


def test_resolve_specs_falls_back_to_legacy_ollama_model(monkeypatch):
    # Back-compat: every existing profiles.json has ollama_model and no model.
    monkeypatch.delenv("CCR_MODEL", raising=False)
    assert p.resolve_specs(None, profile={"ollama_model": "qwen2.5-coder:7b"}) \
        == ["qwen2.5-coder:7b"]


def test_resolve_specs_splits_an_ensemble(monkeypatch):
    monkeypatch.delenv("CCR_MODEL", raising=False)
    assert p.resolve_specs("ollama:a, anthropic:b ,openai:c") == \
        ["ollama:a", "anthropic:b", "openai:c"]


def test_resolve_specs_errors_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("CCR_MODEL", raising=False)
    with pytest.raises(ValueError):
        p.resolve_specs(None, profile={})


# --- reply handling ----------------------------------------------------------

@pytest.mark.parametrize("raw,expect", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('```{"a": 1}```', {"a": 1}),
    ('   {"a": 1}   ', {"a": 1}),
])
def test_parse_json_reply_unwraps_fences(raw, expect):
    # Ollama's format:"json" returns a bare document, but a hosted model asked
    # for JSON in the prompt commonly fences it anyway.
    assert p.parse_json_reply(raw) == expect


def test_parse_json_reply_keeps_the_raw_text_for_salvage():
    # surface_review repairs a TRUNCATED document from exc.content; losing the
    # text would turn a salvageable sample into a discarded one.
    with pytest.raises(p.ReplyParseError) as exc:
        p.parse_json_reply('{"findings": [{"title": "cut off')
    assert exc.value.content.startswith('{"findings"')


def test_reply_parse_error_is_a_value_error():
    # The pipeline's per-sample and per-finding fallbacks catch ValueError.
    assert issubclass(p.ReplyParseError, ValueError)


@pytest.mark.parametrize("cls", [p.TransportError, p.RequestRejected])
def test_backend_errors_are_os_errors(cls):
    # urllib.error.HTTPError -> URLError -> OSError is the shape the pipeline's
    # except tuples were written against. A non-Ollama backend must not slip
    # past them and abort a multi-hour run.
    assert issubclass(cls, OSError)


# --- OllamaBackend envelope + error mapping ----------------------------------

def _body(content):
    class R:
        def __enter__(self_):
            return self_

        def __exit__(self_, *a):
            return False

        def read(self_):
            return json.dumps(content).encode()
    return R()


@pytest.mark.parametrize("body", [
    ["not", "an", "object"],
    {"message": None},
    {"message": {"content": None}},
])
def test_ollama_malformed_envelope_raises_value_error(body):
    # Not AttributeError/TypeError: those are not in the callers' except tuples.
    with pytest.raises(ValueError):
        p.OllamaBackend.content_of(body)


def test_ollama_chat_returns_content_and_counts_usage(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _body(
        {"message": {"content": '{"ok": true}'}, "prompt_eval_count": 11,
         "eval_count": 7}))
    b = p.OllamaBackend("qwen2.5-coder:7b", base_url="localhost:11434")
    assert b.base_url == "http://localhost:11434"          # scheme added
    assert p.parse_json_reply(b.chat([{"role": "user", "content": "hi"}])) == {"ok": True}
    assert b.usage == {"input_tokens": 11, "output_tokens": 7, "calls": 1}


def test_ollama_404_becomes_a_rejection_naming_the_pull(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(p.RequestRejected) as exc:
        p.OllamaBackend("ghost").chat([{"role": "user", "content": "hi"}])
    assert exc.value.code == 404
    assert "ollama pull ghost" in str(exc.value)
    # The distinction that matters: a server that ANSWERED is not "unreachable".
    assert not isinstance(exc.value, p.TransportError)


def test_ollama_refused_connection_becomes_a_transport_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(p.TransportError):
        p.OllamaBackend("m").chat([{"role": "user", "content": "hi"}])


def test_ollama_probe_matches_tag_family(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: _body(
        {"models": [{"name": "qwen2.5-coder:7b"}]}))
    assert p.OllamaBackend("qwen2.5-coder:7b").probe() == (True, True)
    assert p.OllamaBackend("qwen2.5-coder:14b").probe() == (True, True)   # same family
    assert p.OllamaBackend("llama3:8b").probe() == (True, False)


def test_ollama_probe_reports_a_dead_server(monkeypatch):
    def boom(url, timeout=None):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert p.OllamaBackend("m").probe() == (False, False)


def test_make_backend_routes_by_provider():
    assert isinstance(p.make_backend("qwen2.5-coder:14b"), p.OllamaBackend)
    remote = p.make_backend("anthropic:claude-opus-5")
    assert isinstance(remote, p.LangChainBackend) and not remote.is_local


# --- LangChainBackend (driven through a stub client; langchain not required) --

class _StubReply:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage


class _StubClient:
    def __init__(self, reply=None, raises=None):
        self.reply, self.raises, self.seen = reply, raises, None

    def invoke(self, messages):
        self.seen = messages
        if self.raises:
            raise self.raises
        return self.reply


def _lc(monkeypatch, client, spec="anthropic:claude-opus-5"):
    b = p.make_backend(spec)
    monkeypatch.setattr(b, "_client", lambda *a, **k: client)
    return b


def test_langchain_maps_roles_and_returns_text(monkeypatch):
    client = _StubClient(_StubReply('{"ok": true}', {"input_tokens": 3,
                                                     "output_tokens": 4}))
    b = _lc(monkeypatch, client)
    out = b.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert out == '{"ok": true}'
    assert client.seen == [("system", "s"), ("human", "u")]   # "user" -> "human"
    assert b.usage == {"input_tokens": 3, "output_tokens": 4, "calls": 1}


def test_langchain_flattens_block_list_content(monkeypatch):
    # Some providers return typed blocks rather than a string; the text blocks
    # in order are the document we asked for.
    b = _lc(monkeypatch, _StubClient(_StubReply(
        [{"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}])))
    assert p.parse_json_reply(b.chat([{"role": "user", "content": "u"}])) == {"a": 1}


def test_langchain_connection_failure_is_a_transport_error(monkeypatch):
    class APIConnectionError(Exception):
        pass
    b = _lc(monkeypatch, _StubClient(raises=APIConnectionError("no route")))
    with pytest.raises(p.TransportError):
        b.chat([{"role": "user", "content": "u"}])


def test_langchain_other_failures_are_rejections_carrying_the_status(monkeypatch):
    # An auth or rate-limit failure reported as "unreachable" sends debugging
    # at the network when the fix is a key or a backoff.
    class AuthenticationError(Exception):
        status_code = 401
    b = _lc(monkeypatch, _StubClient(raises=AuthenticationError("bad key")))
    with pytest.raises(p.RequestRejected) as exc:
        b.chat([{"role": "user", "content": "u"}])
    assert exc.value.code == 401


def test_langchain_missing_api_key_is_reported_by_name(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(p.RequestRejected) as exc:
        p.make_backend("anthropic:claude-opus-5")._client(0.2)
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_langchain_probe_is_constructability_not_a_billable_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert p.make_backend("openai:gpt-5").probe() == (False, False)


def test_format_usage_reports_across_backends(monkeypatch):
    a, b = p.OllamaBackend("m"), p.OllamaBackend("n")
    a.usage.update(calls=2, input_tokens=100, output_tokens=50)
    b.usage.update(calls=1, input_tokens=10, output_tokens=5)
    assert p.format_usage([a, b]) == "[usage] 3 call(s), 110 input + 55 output tokens"
    assert p.format_usage([p.OllamaBackend("m")]) == ""       # nothing counted


# --- endpoint routing (regression) -------------------------------------------
# --ollama-url always has a value, so callers pass base_url unconditionally.
# Forwarding it to a hosted provider pointed that provider's client at
# localhost:11434 — the request never reached the API, and whatever was
# listening on that port received the code under review.

def test_ollama_url_is_not_forwarded_to_a_hosted_provider():
    b = p.make_backend("anthropic:claude-opus-5", base_url="http://localhost:11434")
    assert b.base_url is None


def test_ollama_url_is_forwarded_to_local_backends():
    assert p.make_backend("qwen:7b", base_url="http://box:11434").base_url \
        == "http://box:11434"
    assert p.make_backend("lc-ollama:qwen:7b", base_url="http://box:11434").base_url \
        == "http://box:11434"


def test_hosted_endpoint_override_comes_from_ccr_api_base(monkeypatch):
    # The escape hatch for an OpenAI-compatible vLLM/TGI server.
    monkeypatch.setenv("CCR_API_BASE", "http://gpu-box:8000/v1")
    assert p.make_backend("openai:qwen", base_url="http://localhost:11434").base_url \
        == "http://gpu-box:8000/v1"


def test_hosted_client_is_built_with_the_right_endpoint(monkeypatch):
    # Constructs a REAL ChatAnthropic: the stubbed-client tests above never
    # reach _client, which is exactly how the base_url bug went unnoticed.
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.delenv("CCR_API_BASE", raising=False)
    b = p.make_backend("anthropic:claude-opus-5", base_url="http://localhost:11434")
    client = b._client(0.3, timeout=1800)
    assert "11434" not in str(getattr(client, "anthropic_api_url", ""))
    assert client.max_tokens == p.DEFAULT_MAX_TOKENS
    assert client.temperature == 0.3


def test_azure_is_not_pre_rejected_for_a_missing_api_key(monkeypatch):
    # Azure authenticates with AAD tokens too; pre-checking a key would reject
    # a valid setup before the SDK ever gets a say.
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert "azure_openai" not in p._PROVIDER_ENV_KEY


# --- concurrency (regression) ------------------------------------------------
# llm_explain drives ONE backend from up to max_workers threads.

def test_usage_counters_survive_concurrent_calls():
    # `d[k] += n` is a read-modify-write and loses updates under the GIL —
    # measured ~19% undercount at 8 threads before the lock. Under-reporting a
    # hosted bill is the one number a caller cannot afford to have wrong.
    import threading
    b = p.OllamaBackend("m")

    def hammer():
        for _ in range(5000):
            b._record_usage({"prompt_eval_count": 1, "eval_count": 2})

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert b.usage == {"calls": 40000, "input_tokens": 40000, "output_tokens": 80000}


def test_concurrent_client_lookups_share_one_client(monkeypatch):
    import threading
    built = []
    b = p.make_backend("anthropic:claude-opus-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def fake_init(model, model_provider=None, **kw):
        obj = object()
        built.append(obj)
        return obj
    monkeypatch.setattr(b, "_init_chat_model", lambda: fake_init)

    seen = []
    threads = [threading.Thread(target=lambda: seen.append(b._client(0.2)))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(id(c) for c in seen)) == 1      # every thread got the same client
