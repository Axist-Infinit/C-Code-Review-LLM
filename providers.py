#!/usr/bin/env python3
"""Provider-agnostic chat backends for the review pipeline.

Neither review lane ever calls a model directly: surface_review.py (whole-file
review) and llm_explain.py (per-finding explanation) both speak JSON-in /
JSON-out to a chat endpoint. This module is the single seam where that endpoint
is chosen, so the same reviewer logic — sampling, critic gate, pooling,
anchor dedup, hallucination validation — runs unchanged against a local Ollama
model or against any model LangChain can reach (Anthropic, OpenAI, Azure,
Bedrock, Google, Groq, an OpenAI-compatible vLLM/TGI server, ...).

Two implementations:

  OllamaBackend     stdlib urllib only. The default, and the air-gapped path:
                    build_airgap_bundle.sh / offline_lockdown.sh must keep
                    working on a box with no third-party packages installed, so
                    nothing in this module imports langchain at import time.
  LangChainBackend  lazily imports langchain.chat_models.init_chat_model, and
                    only when a spec actually names a LangChain provider.

Model specs are "provider:model" strings:

    qwen2.5-coder:14b          -> ollama    (bare tag; back-compat with the
                                             old profiles.json ollama_model)
    ollama:qwen2.5-coder:14b   -> ollama
    anthropic:claude-opus-5    -> langchain
    openai:gpt-5               -> langchain
    lc-ollama:qwen2.5-coder:7b -> langchain, talking to Ollama

Errors are normalised so a caller can tell the three failure modes apart
without knowing which provider ran. The pipeline's retry/salvage logic depends
on that distinction — calling a refused request "unreachable" sends debugging
in exactly the wrong direction:

    TransportError    nothing answered (connection refused, DNS, timeout)
    RequestRejected   the server answered and refused (404 model not pulled,
                      401 bad key, 429 rate limited)
    ReplyParseError   a reply arrived but is not parseable JSON (truncation)
"""
import json
import os
import threading
import urllib.error
import urllib.request

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Providers reachable through LangChain's init_chat_model. Used only to decide
# whether the first colon-separated token of a spec is a provider name or part
# of an Ollama tag: "qwen2.5-coder:14b" must stay one Ollama tag, while
# "anthropic:claude-opus-5" must split. Anything not listed here is treated as
# a bare Ollama tag, which keeps existing configs working untouched.
LANGCHAIN_PROVIDERS = {
    "anthropic", "openai", "azure_openai", "azure_ai", "bedrock",
    "bedrock_converse", "google_anthropic_vertex", "google_genai",
    "google_vertexai", "groq", "cohere", "fireworks", "together", "mistralai",
    "deepseek", "xai", "perplexity", "huggingface", "nvidia", "ibm", "ollama",
}

# Providers whose weights run on this machine. Everything else ships the code
# under review to a third party — which for a C/C++ security reviewer is the
# sensitive artifact, so callers warn before using one.
LOCAL_PROVIDERS = {"ollama", "lc-ollama"}

# A hosted model needs an explicit output cap and the default is usually far
# too small for a full review document (walkthrough + findings + checklist).
DEFAULT_MAX_TOKENS = 8192

# Named so the error message can say which key is missing instead of surfacing
# the provider SDK's own less specific complaint. Listed here ONLY for providers
# where an API key in the environment is the sole auth path — azure_openai
# (AAD tokens), bedrock (AWS credential chain) and vertexai (ADC) all
# authenticate without one, so pre-checking a key would reject a valid setup.
_PROVIDER_ENV_KEY = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "perplexity": "PPLX_API_KEY",
}


# --- error taxonomy ----------------------------------------------------------
# TransportError and RequestRejected both subclass OSError so that the existing
# `except (urllib.error.URLError, OSError, ...)` fallbacks keep catching them:
# urllib.error.HTTPError -> URLError -> OSError was the shape the pipeline was
# written against, and a non-Ollama backend must not slip past those handlers
# and abort a multi-hour run.

class BackendError(Exception):
    """Base for every normalised backend failure."""


class TransportError(BackendError, OSError):
    """Nothing answered: connection refused, DNS failure, socket timeout.

    A dead backend will not heal on retry, so callers stop rather than burn
    the per-sample attempt budget against a closed port.
    """


class RequestRejected(BackendError, OSError):
    """The server ANSWERED and refused the request.

    Distinct from TransportError: the endpoint is up, the request is wrong.
    ``code`` carries the HTTP status when one is available (404 = the Ollama
    tag is not pulled; 401 = bad API key; 429 = rate limited).
    """

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class ReplyParseError(BackendError, ValueError):
    """A reply arrived but is not parseable JSON.

    Distinct from a transport error: the backend WAS reachable, so the caller
    must retry / salvage rather than report "backend unreachable". Carries the
    raw ``content`` so a truncated reply can still be salvaged.
    """

    def __init__(self, message, content=""):
        super().__init__(message)
        self.content = content


# --- spec parsing ------------------------------------------------------------

def parse_spec(spec):
    """Split a model spec into (provider, model).

    Only the FIRST colon is considered, and only when what precedes it is a
    known provider — otherwise the whole string is an Ollama tag. That rule is
    what lets "qwen2.5-coder:14b" and "anthropic:claude-opus-5" coexist without
    the user having to quote or escape anything.

    "lc-ollama:" forces Ollama through LangChain (useful for testing the
    LangChain path against a local server without spending money).
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty model spec")
    head, sep, tail = spec.partition(":")
    head_l = head.strip().lower()
    if sep and tail.strip():
        if head_l == "lc-ollama":
            return "lc-ollama", tail.strip()
        if head_l in LANGCHAIN_PROVIDERS:
            return head_l, tail.strip()
    return "ollama", spec


def is_local(spec):
    """True when the spec's weights run on this machine (nothing leaves it)."""
    provider, _ = parse_spec(spec)
    return provider in LOCAL_PROVIDERS


def normalize_url(url):
    """Ollama's native OLLAMA_HOST is a bare host:port ("127.0.0.1:11434");
    urllib needs a scheme or it raises "unknown url type". Prepend http:// when
    no scheme is present; pass full URLs through unchanged."""
    if url and "://" not in url:
        return "http://" + url
    return url


def strip_code_fence(text):
    """Drop a ```json ... ``` wrapper if the model added one.

    Ollama's format:"json" guarantees a bare JSON document, but a hosted model
    asked for JSON in the prompt often fences it anyway. Cheap to undo here,
    and it keeps the fence out of every downstream JSON parse.
    """
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    nl = s.find("\n")
    if nl == -1:                       # single-line fence: ```{"a": 1}```
        return s[:-3].strip() if s.rstrip().endswith("```") else s.strip()
    first, rest = s[:nl].strip().lower(), s[nl + 1:]
    if first and first not in ("json", "javascript", "js"):
        # Not a language tag we recognise — leave the body untouched rather
        # than guess where the fence really ended.
        return ("```" + s).strip()
    end = rest.rfind("```")
    return (rest[:end] if end != -1 else rest).strip()


def parse_json_reply(content):
    """Parse a model reply as JSON, raising ReplyParseError with the raw text.

    Keeping the raw text on the exception is what lets surface_review salvage a
    truncated document instead of discarding a whole sample.
    """
    text = strip_code_fence(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ReplyParseError(f"{e} [reply was {len(text)} chars]", text) from e


# --- backends ----------------------------------------------------------------

class ChatBackend:
    """One chat endpoint bound to one model.

    Subclasses implement chat() and probe(). chat() returns the model's reply
    as RAW TEXT — parsing is the caller's job, because the two lanes want
    different things from a malformed reply (surface_review salvages truncated
    documents; llm_explain falls back to the regex heuristic per finding).
    """

    provider = "unset"

    def __init__(self, model, spec=None):
        self.model = model
        self.spec = spec or f"{self.provider}:{model}"
        # {input_tokens, output_tokens} accumulated across calls; a hosted
        # provider bills on these, so a run must be able to report them.
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        # llm_explain drives ONE backend from up to max_workers threads, and
        # `d[k] += n` is a read-modify-write that loses updates under the GIL —
        # measured ~19% undercount at 8 threads. Under-reporting a hosted bill
        # is exactly the number the caller cannot afford to have wrong.
        self._usage_lock = threading.Lock()

    def _add_usage(self, input_tokens=0, output_tokens=0):
        with self._usage_lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += int(input_tokens or 0)
            self.usage["output_tokens"] += int(output_tokens or 0)

    @property
    def is_local(self):
        return self.provider in LOCAL_PROVIDERS

    def chat(self, messages, *, temperature=0.2, num_ctx=None, timeout=600,
             json_mode=True, schema=None):
        raise NotImplementedError

    def chat_json(self, messages, *, schema=None, temperature=0.2, num_ctx=None,
                  timeout=600):
        """Chat and return the reply PARSED as an object.

        The base implementation goes through chat() and parses the text, which
        is what keeps ReplyParseError carrying the raw reply — surface_review
        salvages a truncated document from it, and losing that would turn a
        recoverable sample into a discarded one. A backend with provider-native
        structured output overrides this, and must preserve the same contract.
        """
        return parse_json_reply(self.chat(
            messages, schema=schema, temperature=temperature, num_ctx=num_ctx,
            timeout=timeout))

    def probe(self):
        """Return (reachable, model_present) without doing billable work."""
        raise NotImplementedError

    def describe(self):
        return self.spec

    def __repr__(self):
        return f"<{type(self).__name__} {self.spec}>"


class OllamaBackend(ChatBackend):
    """Ollama /api/chat over stdlib urllib — no third-party imports.

    This is the air-gapped path: the offline bundle installs no packages beyond
    the pinned requirements, so this backend must keep working with nothing but
    the standard library available.
    """

    provider = "ollama"

    def __init__(self, model, base_url=None, spec=None):
        super().__init__(model, spec=spec or f"ollama:{model}")
        self.base_url = normalize_url(
            base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL)
        # Latched on the first HTTP 400 for a schema `format`, so an old server
        # costs one rejected request per run rather than one per call.
        self._schema_unsupported = False

    @staticmethod
    def content_of(resp):
        """Extract the reply text from an /api/chat body.

        Ollama normally returns {"message": {"content": "<text>"}}, but on some
        errors/timeouts it returns {"message": null} or a non-object body. A
        bare resp["message"]["content"] then raises AttributeError/TypeError,
        which is not in the callers' except tuples and would abort an entire
        multi-hour run. Raise ValueError instead so the per-sample and
        per-finding fallbacks handle it.
        """
        if not isinstance(resp, dict):
            raise ValueError(f"non-object chat response body: {type(resp).__name__}")
        message = resp.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"chat response 'message' is {type(message).__name__}, "
                             f"not an object")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"chat response 'content' is {type(content).__name__}, "
                             f"not a string")
        return content

    def chat(self, messages, *, temperature=0.2, num_ctx=None, timeout=600,
             json_mode=True, schema=None):
        # Ollama >= 0.5 accepts a JSON Schema as `format` and constrains
        # sampling to it. An older server rejects the object with HTTP 400; we
        # degrade to plain JSON mode once, remember it, and carry on — a
        # structured-output upgrade must never break a working install.
        use_schema = schema is not None and not self._schema_unsupported
        try:
            return self._post(messages, temperature, num_ctx, timeout,
                              schema if use_schema else ("json" if json_mode else None))
        except RequestRejected as rj:
            if not use_schema or rj.code != 400:
                raise
            self._schema_unsupported = True
            print(f"[warn] {self.base_url} rejected a JSON-Schema format "
                  f"(needs Ollama >= 0.5); falling back to plain JSON mode")
        return self._post(messages, temperature, num_ctx, timeout,
                          "json" if json_mode else None)

    def _post(self, messages, temperature, num_ctx, timeout, fmt):
        options = {"temperature": temperature}
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": options,
        }
        if fmt is not None:
            payload["format"] = fmt
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as he:
            # HTTPError is a URLError subclass, so this must come first.
            detail = (f"model '{self.model}' is not pulled on that server "
                      f"(ollama pull {self.model})") if he.code == 404 else str(he)
            raise RequestRejected(detail, code=he.code) from he
        except (urllib.error.URLError, OSError) as te:
            raise TransportError(f"{self.base_url}: {te}") from te
        content = self.content_of(resp)
        self._record_usage(resp)
        return content

    def _record_usage(self, resp):
        resp = resp if isinstance(resp, dict) else {}
        self._add_usage(resp.get("prompt_eval_count"), resp.get("eval_count"))

    def probe(self, timeout=5):
        """Return (reachable, model_present) by listing /api/tags."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=timeout) as r:
                tags = json.load(r)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False, False
        names = {m.get("name", "") for m in (tags.get("models") or [])}
        present = any(n == self.model or n.split(":")[0] == self.model.split(":")[0]
                      for n in names)
        return True, present


class LangChainBackend(ChatBackend):
    """Any provider LangChain's init_chat_model can construct.

    The client is built lazily and cached per temperature: temperature is a
    constructor field on most chat-model integrations, and surface_review walks
    it upward across retry attempts to break a degenerate repetition loop. A
    handful of cached clients is cheaper and far more portable than trying to
    push a per-call override through every integration.

    JSON mode is only requested for Ollama, where the flag is unambiguous. For
    hosted providers the system prompts already demand a bare JSON object and
    strip_code_fence() handles the common deviation; provider-native structured
    output (tool-calling schemas) is a follow-up, not a prerequisite.
    """

    def __init__(self, provider, model, spec=None, base_url=None,
                 max_tokens=DEFAULT_MAX_TOKENS, **model_kwargs):
        self.provider = provider
        super().__init__(model, spec=spec or f"{provider}:{model}")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.model_kwargs = model_kwargs
        self._clients = {}
        # Guards the client cache: the explainer builds its chat_fn once and
        # calls it from every worker thread, so an unguarded cache would
        # construct a duplicate SDK client per concurrent miss.
        self._clients_lock = threading.Lock()
        # Latched when an integration turns out not to implement structured
        # output, so the fallback costs one failed attempt per run, not per call.
        self._structured_unsupported = False

    @property
    def _lc_provider(self):
        """The provider name init_chat_model understands."""
        return "ollama" if self.provider == "lc-ollama" else self.provider

    def _init_chat_model(self):
        try:
            from langchain.chat_models import init_chat_model
        except ImportError:
            try:                              # older layouts kept it here
                from langchain_core.language_models import init_chat_model
            except ImportError as e:
                # A missing package is a CONFIG failure, not an unreachable
                # backend: reporting "unreachable" would send debugging at the
                # network when the fix is a pip install.
                raise RequestRejected(
                    f"provider '{self.provider}' needs LangChain: "
                    f"pip install -r requirements-langchain.txt "
                    f"(and the provider package, e.g. "
                    f"langchain-{self._lc_provider.replace('_', '-')})") from e
        return init_chat_model

    def _client(self, temperature, num_ctx=None, timeout=None, json_mode=True):
        key = (temperature, num_ctx, timeout, json_mode)
        with self._clients_lock:
            if key in self._clients:
                return self._clients[key]
        init_chat_model = self._init_chat_model()
        kwargs = dict(self.model_kwargs)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self._lc_provider == "ollama":
            # Local: no output cap needed, but the KV-cache size and JSON mode
            # are the two settings that decide whether a local reply is usable.
            if num_ctx:
                kwargs["num_ctx"] = num_ctx
            if json_mode:
                kwargs.setdefault("format", "json")
        else:
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if timeout:
                kwargs["timeout"] = timeout
        env_key = _PROVIDER_ENV_KEY.get(self._lc_provider)
        if env_key and not os.environ.get(env_key):
            raise RequestRejected(
                f"provider '{self.provider}' needs ${env_key} in the environment")
        try:
            client = init_chat_model(self.model, model_provider=self._lc_provider,
                                     **kwargs)
        except ImportError as e:
            raise RequestRejected(
                f"provider '{self.provider}' needs its LangChain integration "
                f"package installed: {e}") from e
        except Exception as e:                # pragma: no cover - provider-specific
            raise RequestRejected(
                f"could not construct {self.spec}: {type(e).__name__}: {e}") from e
        with self._clients_lock:
            # A racing thread may have won; either client is equivalent, so keep
            # whichever landed first and let this one go.
            return self._clients.setdefault(key, client)

    @staticmethod
    def _to_lc_messages(messages):
        """Map our {"role", "content"} dicts to LangChain's role names."""
        role_map = {"user": "human", "assistant": "ai", "system": "system",
                    "human": "human", "ai": "ai"}
        out = []
        for m in messages:
            role = role_map.get(str(m.get("role", "user")).lower(), "human")
            out.append((role, m.get("content", "")))
        return out

    @staticmethod
    def _content_of(reply):
        """Flatten an AIMessage's content to text.

        Some providers return a list of typed blocks rather than a string; the
        text blocks concatenated in order are the document we asked for.
        """
        content = getattr(reply, "content", reply)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)
        raise ValueError(f"reply content is {type(content).__name__}, not text")

    # Exception CLASS NAMES that mean "nothing answered". Matched by name so
    # this module never has to import a provider SDK to classify its errors.
    # Anything unmatched is reported as a rejection carrying the provider's own
    # message, which is more useful than a wrong "unreachable".
    _TRANSPORT_NAMES = {
        "APIConnectionError", "APITimeoutError", "ConnectionError",
        "ConnectTimeout", "ConnectError", "ReadTimeout", "Timeout",
        "TimeoutException", "TimeoutError", "ServiceUnavailableError",
    }

    def chat(self, messages, *, temperature=0.2, num_ctx=None, timeout=600,
             json_mode=True, schema=None):
        client = self._client(temperature, num_ctx=num_ctx, timeout=timeout,
                              json_mode=json_mode)
        reply = self._invoke(client, messages)
        self._record_usage(reply)
        return self._content_of(reply)

    def chat_json(self, messages, *, schema=None, temperature=0.2, num_ctx=None,
                  timeout=600):
        """Use the provider's own structured-output mode when a schema is given.

        with_structured_output routes the schema to tool-calling / json_schema
        mode, so the shape is enforced by the provider rather than by asking
        nicely. include_raw=True is essential and not a detail: without it the
        raw AIMessage is discarded, which would lose BOTH the token usage this
        run reports and the raw text that surface_review salvages a truncated
        document from.

        Falls back to plain text + parse for any provider whose integration
        does not implement structured output.
        """
        if schema is None or self._structured_unsupported:
            return super().chat_json(messages, schema=schema, temperature=temperature,
                                     num_ctx=num_ctx, timeout=timeout)
        client = self._client(temperature, num_ctx=num_ctx, timeout=timeout,
                              json_mode=True)
        try:
            structured = client.with_structured_output(schema, include_raw=True)
        except (NotImplementedError, TypeError, ValueError) as e:
            self._structured_unsupported = True
            print(f"[warn] {self.spec} does not support structured output "
                  f"({type(e).__name__}); falling back to JSON-mode parsing")
            return super().chat_json(messages, schema=schema, temperature=temperature,
                                     num_ctx=num_ctx, timeout=timeout)

        result = self._invoke(structured, messages)
        raw = result.get("raw") if isinstance(result, dict) else None
        if raw is not None:
            self._record_usage(raw)
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if isinstance(parsed, dict):
            return parsed
        if parsed is not None and hasattr(parsed, "model_dump"):   # a Pydantic schema
            return parsed.model_dump()
        # Structured output failed to produce an object. Surface it as a parse
        # error carrying whatever text did arrive, so the caller's existing
        # retry-then-salvage path applies exactly as it does for Ollama.
        err = (result.get("parsing_error") if isinstance(result, dict) else None) or \
            f"structured output returned {type(parsed).__name__}"
        text = self._content_of(raw) if raw is not None else ""
        raise ReplyParseError(f"{self.spec}: {err}", text)

    def _invoke(self, client, messages):
        try:
            return client.invoke(self._to_lc_messages(messages))
        except BackendError:
            raise
        except Exception as e:
            name = type(e).__name__
            if name in self._TRANSPORT_NAMES or isinstance(e, (TimeoutError, ConnectionError)):
                raise TransportError(f"{self.spec}: {name}: {e}") from e
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            raise RequestRejected(f"{self.spec}: {name}: {e}",
                                  code=code if isinstance(code, int) else None) from e

    def _record_usage(self, reply):
        meta = getattr(reply, "usage_metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        self._add_usage(meta.get("input_tokens"), meta.get("output_tokens"))

    def probe(self, timeout=5):
        """Constructability check only — a hosted probe would be billable."""
        try:
            self._client(None)
        except BackendError:
            return False, False
        return True, True


def make_backend(spec, *, base_url=None, max_tokens=DEFAULT_MAX_TOKENS, **kwargs):
    """Build the backend named by a model spec. See parse_spec() for the grammar.

    ``base_url`` is the OLLAMA endpoint and is forwarded ONLY to local backends.
    Callers pass it unconditionally (it comes from --ollama-url, which always
    has a value), so forwarding it to a hosted provider would silently point
    that provider's client at localhost:11434. A non-local endpoint override —
    for an OpenAI-compatible vLLM/TGI server, say — comes from $CCR_API_BASE
    instead, which is only set when it is actually meant.
    """
    provider, model = parse_spec(spec)
    if provider == "ollama":
        return OllamaBackend(model, base_url=base_url, spec=f"ollama:{model}")
    lc_base = base_url if provider in LOCAL_PROVIDERS else os.environ.get("CCR_API_BASE")
    return LangChainBackend(provider, model, spec=f"{provider}:{model}",
                            base_url=lc_base or None, max_tokens=max_tokens, **kwargs)


def resolve_specs(explicit=None, profile=None, env_var="CCR_MODEL"):
    """Pick the model spec(s) to run, in precedence order.

    1. an explicit CLI value (comma-separated for an ensemble)
    2. $CCR_MODEL
    3. the profile's "model" key
    4. the profile's legacy "ollama_model" key

    Step 4 is what keeps every existing profiles.json and install working after
    this change: a bare tag parses back to the Ollama provider.
    """
    raw = explicit or os.environ.get(env_var) or ""
    if not raw and profile:
        raw = profile.get("model") or profile.get("ollama_model") or ""
    specs = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not specs:
        raise ValueError("no model spec: pass --model, set $CCR_MODEL, or give "
                         "the profile a 'model' key")
    return specs


def egress_warning(specs):
    """The one-line warning for specs that send code off this machine, or ""..

    The tool's whole premise is reviewing C/C++ that is too sensitive to hand
    around, so a remote provider is worth stating out loud rather than
    discovering in a proxy log.
    """
    remote = sorted({s for s in specs if not is_local(s)})
    if not remote:
        return ""
    return (f"[egress] {', '.join(remote)} is a REMOTE provider — the source "
            f"under review is sent off this machine. Use a local ollama: spec "
            f"to keep the review offline.")


def format_usage(backends):
    """One-line token summary across backends, or "" if nothing was counted."""
    calls = sum(b.usage["calls"] for b in backends)
    inp = sum(b.usage["input_tokens"] for b in backends)
    out = sum(b.usage["output_tokens"] for b in backends)
    if not calls or not (inp or out):
        return ""
    return f"[usage] {calls} call(s), {inp} input + {out} output tokens"
