# Model providers — running the reviewer on an existing model

The review pipeline does not contain a model. Both LLM lanes speak JSON-in /
JSON-out to a chat endpoint, so the reviewer's actual logic — multi-sample
pooling, the completeness critic and its regression gate, anchor dedup and
topic consolidation, CWE canonicalisation, the mechanical false-positive gate
against the real source — is independent of which model answers.

`providers.py` is the single seam where that endpoint is chosen. Everything
else is unchanged whether the reply came from a 7B running on your laptop or
from a frontier model over an API.

```
surface_review.py ─┐                        ┌─ OllamaBackend    (stdlib urllib)
                   ├─ providers.py ─────────┤
llm_explain.py    ─┘   make_backend(spec)   └─ LangChainBackend (init_chat_model)
                                                 anthropic / openai / azure /
                                                 bedrock / google / groq /
                                                 vLLM (openai-compatible) / ...
```

## Model specs

A model is named by a `provider:model` spec:

| Spec | Goes to |
|---|---|
| `qwen2.5-coder:14b` | Ollama — a **bare tag still means Ollama**, so nothing that worked before changes |
| `ollama:qwen2.5-coder:32b` | Ollama, explicitly |
| `anthropic:claude-opus-5` | LangChain → Anthropic |
| `openai:gpt-5` | LangChain → OpenAI |
| `lc-ollama:qwen2.5-coder:7b` | LangChain → your local Ollama (exercises the LangChain path without spending anything) |

Only the first colon is considered, and only when what precedes it names a
known provider — which is what lets `qwen2.5-coder:14b` and
`anthropic:claude-opus-5` coexist without quoting. An unrecognised prefix is
treated as an Ollama tag, so `deepseek-coder-v2:16b` stays one tag rather than
being sent to DeepSeek's API as a model called `16b`.

Where the spec comes from, highest precedence first:

1. `--model` / `--models` on the command line
2. `$CCR_MODEL`
3. the profile's `model` key in `profiles.json`
4. the profile's `ollama_model` key

Shipped profiles set only `ollama_model`, so an existing install keeps running
its local model with no configuration. Add `model` to a profile when you want a
different default; it wins over `ollama_model` for that profile.

## Setup

The default Ollama path needs nothing new — it is stdlib-only on purpose, so
the air-gapped bundle keeps working on a box with no packages installed, and
`providers.py` never imports langchain unless a spec asks for it.

For any other provider:

```bash
pip install -r requirements-langchain.txt
pip install langchain-anthropic          # the integration for the provider you use
export ANTHROPIC_API_KEY=...
```

`--ollama-url` is only ever sent to a local backend. To point an
OpenAI-compatible endpoint (vLLM, TGI, a gateway) somewhere else, set
`$CCR_API_BASE`:

```bash
CCR_API_BASE=http://gpu-box:8000/v1 OPENAI_API_KEY=x \
    python surface_review.py src/*.c --model openai:qwen2.5-coder-32b --allow-remote
```

## Running

```bash
# Unchanged: the local model named by the detected profile
python surface_review.py src/*.c --md report.md

# A hosted model
python surface_review.py src/*.c --md report.md \
    --model anthropic:claude-opus-5 --allow-remote

# Per-role models: a strong reviewer, a DIFFERENT model as critic. A
# cross-provider critic de-hallucinates better than a model critiquing itself.
python surface_review.py src/*.c --md report.md --allow-remote \
    --models anthropic:claude-opus-5 --critic-model openai:gpt-5

# Ensemble across providers, pooled and consolidated as usual
python surface_review.py src/*.c --md report.md --allow-remote \
    --models "ollama:qwen2.5-coder:32b,anthropic:claude-opus-5" --samples 2

# The per-finding explainer takes the same specs
python llm_explain.py classifier_findings.json --out llm_findings.json \
    --model anthropic:claude-opus-5 --allow-remote
```

## `--allow-remote` is required, deliberately

This tool's premise is reviewing C/C++ that is too sensitive to hand around,
and the code under review is precisely the sensitive artifact. A remote
provider sends it off the machine, so it is never reachable by accepting a
default: without `--allow-remote` the run stops and tells you what it would
have sent where. Local specs (`ollama:`, `lc-ollama:`) never need the flag.

Runs that do use a remote provider print an `[egress]` line, and every run
prints a `[usage]` token total so a hosted bill is visible in the output rather
than discovered later.

## Cost and latency

A whole-module review is a long context and a long reply, and the pipeline
multiplies that by samples × models × the critic pass. Before pointing
`--samples 3 --models a,b` at a hosted API, note that this is 6 full-context
requests plus a critic per file group. `--max-tokens` caps the reply (hosted
providers only; too small a cap truncates the review document), and the
`[usage]` line reports what a run actually spent.

## Failure modes

Backends normalise their errors into three classes so the pipeline's retry and
salvage logic works the same for every provider, and so a diagnosis points at
the real cause:

| Class | Means | Pipeline behaviour |
|---|---|---|
| `TransportError` | nothing answered (refused, DNS, socket timeout) | stop — it will not heal on retry |
| `RequestRejected` | the endpoint answered and refused: 404 tag not pulled, 401 bad key, 429 rate limited, missing integration package | stop, reporting the provider's own reason |
| `ReplyParseError` | a reply arrived but is not parseable JSON | retry with a nudged temperature, then salvage the complete prefix |

That distinction is load-bearing: reporting a refused request as "unreachable"
sends debugging at the network when the fix is a `pip install`, an API key, or
an `ollama pull`.

## What this means for the training stack

Nothing here deletes it. `model/train_*.py` and the LoRA lane remain the way to
specialise a local model for offline use.

What changes is what `corpus/reviews/` is *for*. With frozen models in play,
those Claude-authored reviews plus `score_review.py`, `evaluate_model.py` and
the eval manifest stop being only training data and become a **benchmark** —
the way to measure whether a hosted model, a local model, or your fine-tune
actually reviews C better against your own rubric.

## Not yet done

- **Provider-native structured output.** `surface_review._repair_truncated_json`
  exists because a local model in JSON mode stops mid-document. Tool-calling
  providers can be given the review schema directly via
  `with_structured_output`, which removes most of that failure class. The
  repair path stays regardless — it is still load-bearing for Ollama.
- **Response caching.** A LangChain SQLite cache would make re-reviewing an
  unchanged file free.
- **Rate-limit-aware concurrency.** `llm_explain` sizes its thread pool from the
  profile's `max_workers`, which is a local-GPU quantity; a hosted provider
  wants a limit set by its rate limit instead.
- **`ccr.sh` menu.** The menu already passes its model tag through as `--model`,
  so specs work from it — but it has no `--allow-remote` toggle yet, so a remote
  spec entered there stops with the egress message. Use the CLI for remote
  providers until the menu grows the toggle.
- **`compare_review.py`** still calls the Anthropic SDK directly. It is a
  side-by-side comparison tier rather than part of the review path, so it was
  left alone; routing it through `providers.py` would let it compare any two
  providers instead of local-vs-Claude.
