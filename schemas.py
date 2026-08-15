#!/usr/bin/env python3
"""JSON Schemas for the two model replies, for provider-native structured output.

The prompts in surface_review.py / llm_explain.py stay exactly as they are: their
field descriptions are the reviewer's actual guidance ("exploitation: ... say
what an attacker actually does and what they end up with"), and no schema can
carry that. What a schema adds is ENFORCEMENT of the shape the prompt asks for,
at the decoding layer rather than by hoping:

  * Ollama >= 0.5 takes a JSON Schema as `format` and constrains sampling to it.
  * LangChain's with_structured_output() routes it to the provider's own
    tool-calling / json_schema mode.

Two failures this addresses, both already documented in this repo:

  1. Dropped keys. The v2 splice comment records "exploitation" being emitted
     0/7 times on qwen2.5-coder:14b when it was described late in the prompt.
     A required key in the schema is enforced by the decoder, not by prompt
     placement.
  2. Truncation. _repair_truncated_json exists because a local model falls into
     degenerate repetition mid-document. Schema-constrained decoding cannot
     remove that entirely, but it removes the "valid JSON, wrong shape" class
     around it. The repair path stays regardless — it is still load-bearing.

Deliberately PERMISSIVE beyond those required keys: `additionalProperties` is
left open and optional fields stay optional, because the pipeline already
tolerates missing keys (validate_findings, canonicalize_cwes, normalize_severity
and the pooling code all backfill), and an over-tight schema would make a model
fail a whole sample rather than return a usable partial one.

Plain dicts rather than Pydantic models: the Ollama backend must stay
stdlib-only for the air-gapped bundle, and both consumers accept a JSON Schema
dict directly.
"""

SEVERITIES = ["critical", "high", "medium", "low", "info"]

# A walkthrough step: what_the_code_does / what_could_go_wrong are ORDERED
# ARRAYS of these. Modelled as objects because a model that returns a bare
# string here is the single most common shape drift (render_markdown has a
# _walkthrough_md tolerance for it, which this makes unnecessary).
_STEP = {
    "type": "object",
    "properties": {
        "file": {"type": "string"},
        "lines": {"type": "string"},
        "code": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["lines", "explanation"],
}

_ANCHOR = {
    "type": "object",
    "properties": {
        "anchor": {"type": "string"},
        "disposition": {"type": "string", "enum": ["finding", "dismissed"]},
        "reason": {"type": "string"},
    },
    "required": ["anchor", "disposition"],
}

# Required per finding: the fields the report and the SARIF/PR annotators cannot
# render without, plus the CWE that canonicalize_cwes corrects rather than
# invents. Everything else is optional and backfilled downstream.
_FINDING_REQUIRED_V1 = ["title", "anchor", "file", "line", "cwe", "severity",
                        "failure_mode"]


def _finding(version):
    props = {
        "title": {"type": "string"},
        "anchor": {"type": "string"},
        "file": {"type": "string"},
        "line": {"type": "string"},
        "code": {"type": "string"},
        "bug_class": {"type": "string"},
        "cwe": {"type": "string"},
        "severity": {"type": "string", "enum": SEVERITIES},
        "where_it_lives": {"type": "string"},
        "invariant": {"type": "string"},
        "failure_mode": {"type": "string"},
        "cve_analog": {"type": "string"},
        "what_to_confirm": {"type": "string"},
    }
    required = list(_FINDING_REQUIRED_V1)
    if int(version) >= 2:
        props["exploitation"] = {"type": "string"}
        props["fix"] = {"type": "string"}
        # The prompt says REQUIRED, never omit — so require them here too. This
        # is the enforcement the "0/7 emitted" comment in surface_review.py was
        # working around by moving the description earlier in the prompt.
        required += ["exploitation", "fix"]
    return {"type": "object", "properties": props, "required": required}


def review_schema(version=2):
    """The whole-file review document (surface_review.py).

    Used for the pass-1 review, the critic pass and the LLM consolidation pass:
    all three return the SAME document shape, which is exactly why the critic
    was able to silently drop v2 keys before.
    """
    props = {
        "subsystem": {"type": "string"},
        "provenance": {"type": "string"},
        "trust_boundary": {"type": "string"},
        "what_the_code_does": {"type": "array", "items": _STEP},
        "what_could_go_wrong": {"type": "array", "items": _STEP},
        "summary": {"type": "string"},
        "reviewed_anchors": {"type": "array", "items": _ANCHOR},
        "findings": {"type": "array", "items": _finding(version)},
        "audit_checklist": {"type": "array", "items": {"type": "string"}},
    }
    required = ["subsystem", "what_the_code_does", "what_could_go_wrong",
                "summary", "reviewed_anchors", "findings"]
    if int(version) >= 2:
        # NOT required: unlike the per-finding "exploitation"/"fix", the prompt
        # describes these without marking them REQUIRED, and render_markdown
        # already omits the v2 sections when they are absent. The schema
        # enforces what the prompt declares required and nothing beyond it —
        # every extra required key is a way for a usable reply to be rejected.
        props["lesson"] = {"type": "string"}
        props["secondary_observations"] = {"type": "array",
                                           "items": {"type": "string"}}
    return {"title": "SecurityReview", "type": "object",
            "properties": props, "required": required}


# One per-finding explanation (llm_explain.py). The severity enum matters here:
# normalize_severity currently repairs an out-of-vocabulary value after the
# fact, and an allow-listed enum stops it being emitted at all — which keeps
# garbage out of SARIF ruleIds without relying on the repair.
EXPLAIN_SCHEMA = {
    "title": "FindingExplanation",
    "type": "object",
    "properties": {
        "is_vulnerable": {"type": "boolean"},
        "issue": {"type": "string"},
        "cwe": {"type": "string"},
        "severity": {"type": "string", "enum": SEVERITIES},
        "what_code_does": {"type": "string"},
        "what_could_go_wrong": {"type": "string"},
        "vulnerability": {"type": "string"},
        "explanation": {"type": "string"},
        "fix": {"type": "string"},
    },
    "required": ["is_vulnerable", "issue", "cwe", "severity", "what_code_does",
                 "what_could_go_wrong", "explanation"],
}
