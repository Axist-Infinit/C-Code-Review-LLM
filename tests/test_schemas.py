"""Tests for schemas.py — the JSON Schemas sent for structured output.

These pin the contract in both directions: the schema must require what the
prompt calls REQUIRED (that is the whole point of enforcing it at the decoder),
and it must NOT require anything the pipeline is happy to backfill, because an
over-tight schema turns a usable partial reply into a failed sample.
"""
import json

import pytest

import schemas
import surface_review as sr


def test_review_schema_tracks_the_prompt_version():
    # surface_review builds its prompt from SCHEMA_VERSION; the schema it sends
    # must describe the same document, or the model is asked for one shape and
    # constrained to another.
    assert sr.review_json_schema() == schemas.review_schema(sr.SCHEMA_VERSION)


def test_v2_finding_fields_are_required():
    # The measured failure this addresses: surface_review's splice comment
    # records "exploitation" emitted 0/7 times when described late in the
    # prompt. Prompt placement is a workaround; a required key is enforcement.
    finding = schemas.review_schema(2)["properties"]["findings"]["items"]
    for field in sr.SCHEMA_V2_FINDING_FIELDS:
        assert field in finding["properties"]
        assert field in finding["required"]


def test_v2_top_fields_are_present_but_not_required():
    # Unlike the per-finding v2 keys, the prompt describes these WITHOUT marking
    # them REQUIRED, and render_markdown omits the v2 sections when absent. The
    # schema must not require more than the prompt does: every extra required
    # key is another way for a usable reply to be rejected outright.
    schema = schemas.review_schema(2)
    for field in sr.SCHEMA_V2_TOP_FIELDS:
        assert field in schema["properties"]
        assert field not in schema["required"]


def test_v1_schema_omits_the_v2_fields():
    # The SFT builder reproduces the v1 prompt exactly to match its teacher
    # corpus; a v1 run must not be constrained to v2 keys.
    finding = schemas.review_schema(1)["properties"]["findings"]["items"]
    for field in sr.SCHEMA_V2_FINDING_FIELDS:
        assert field not in finding["properties"]
    assert "lesson" not in schemas.review_schema(1)["properties"]


def test_severity_enum_matches_the_pipelines_allow_list():
    # normalize_severity repairs an out-of-vocabulary severity after the fact;
    # the enum stops it being emitted. They must agree, or the schema would
    # permit a value the pipeline then rewrites.
    import llm_explain
    for sev in schemas.SEVERITIES:
        assert llm_explain.normalize_severity(sev) == sev
    assert schemas.EXPLAIN_SCHEMA["properties"]["severity"]["enum"] == schemas.SEVERITIES


def test_walkthrough_steps_are_objects_not_strings():
    # A bare string here is the most common shape drift; the renderers carry a
    # tolerance for it, which the schema makes unnecessary going forward.
    step = schemas.review_schema(2)["properties"]["what_the_code_does"]["items"]
    assert step["type"] == "object"
    assert set(step["required"]) == {"lines", "explanation"}


def test_optional_fields_stay_optional():
    # The pipeline backfills these (canonicalize_cwes, validate_findings,
    # pool_samples). Requiring them would fail a whole sample over a field
    # nothing downstream needs.
    finding = schemas.review_schema(2)["properties"]["findings"]["items"]
    for field in ("cve_analog", "code", "bug_class", "where_it_lives",
                  "invariant", "what_to_confirm"):
        assert field in finding["properties"]
        assert field not in finding["required"]
    assert "audit_checklist" not in schemas.review_schema(2)["required"]


def test_schemas_are_json_serialisable():
    # They travel as JSON in the Ollama request body.
    json.dumps(schemas.review_schema(2))
    json.dumps(schemas.EXPLAIN_SCHEMA)


@pytest.mark.parametrize("version", [1, 2])
def test_a_good_review_satisfies_its_own_schema(version):
    """The schema must accept a review the pipeline already considers valid.

    Guards the real risk of tightening a schema: constraining the decoder to
    something the rest of the code would have rejected anyway.
    """
    jsonschema = pytest.importorskip("jsonschema")
    finding = {"title": "unbounded copy", "anchor": "strcpy call", "file": "x.c",
               "line": "3", "cwe": "CWE-120", "severity": "high",
               "bug_class": "buffer copy", "failure_mode": "overflow",
               "invariant": "bound the copy", "what_to_confirm": "check dst size",
               "where_it_lives": "x.c", "cve_analog": ""}
    review = {
        "subsystem": "demo", "summary": "s",
        "what_the_code_does": [{"file": "x.c", "lines": "1", "code": "strcpy(b, s);",
                                "explanation": "copies"}],
        "what_could_go_wrong": [{"file": "x.c", "lines": "1", "code": "strcpy(b, s);",
                                 "explanation": "overflows"}],
        "reviewed_anchors": [{"anchor": "strcpy call", "disposition": "finding"}],
        "findings": [finding], "audit_checklist": ["check strcpy"],
    }
    if version >= 2:
        finding.update(exploitation="overwrites the return address", fix="use strlcpy")
        review.update(lesson="validate length with the buffer", secondary_observations=[])
    jsonschema.validate(review, schemas.review_schema(version))


def test_a_finding_missing_exploitation_is_rejected_by_v2():
    jsonschema = pytest.importorskip("jsonschema")
    review = {
        "subsystem": "d", "summary": "s", "lesson": "l",
        "what_the_code_does": [], "what_could_go_wrong": [], "reviewed_anchors": [],
        "findings": [{"title": "t", "anchor": "a", "file": "x.c", "line": "1",
                      "cwe": "CWE-120", "severity": "high", "failure_mode": "f",
                      "fix": "patch"}],          # no "exploitation"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(review, schemas.review_schema(2))


def test_the_teacher_corpus_validates_against_its_own_schema_version():
    """The 325 Claude-authored reviews in corpus/reviews/ must satisfy the
    schema for the version they were written at.

    This is the real guard against over-tightening: if a schema change would
    reject the reviews this project treats as the quality bar, it would reject
    good model output too. Skips rather than fails when the corpus is absent
    (the air-gapped bundle ships without it).
    """
    import glob
    import os
    import sys
    jsonschema = pytest.importorskip("jsonschema")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(repo_root, "corpus", "reviews", "train")
    files = sorted(glob.glob(os.path.join(root, "*.claude.json")))
    if not files:
        pytest.skip("teacher corpus not present")

    # model/build_surface_sft.py imports its sibling as a top-level module, so
    # model/ has to be importable the way the script itself is run.
    import importlib
    model_dir = os.path.join(repo_root, "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    try:
        build_sft = importlib.import_module("model.build_surface_sft")
    except ImportError as e:                      # trainer-side deps absent
        pytest.skip(f"SFT builder unimportable: {e}")
    docs = []
    for path in files:
        doc = json.load(open(path, encoding="utf-8"))
        docs.append(doc.get("review", doc) if isinstance(doc, dict) else doc)
    # Ask the SFT builder which version this corpus is, rather than assuming:
    # it is the same detection the trainer uses to pick its prompt.
    version = build_sft.corpus_schema_version(docs)[0]
    schema = schemas.review_schema(version)
    for path, doc in zip(files, docs):
        try:
            jsonschema.validate(doc, schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"{os.path.basename(path)} fails the v{version} schema at "
                        f"{list(e.absolute_path)}: {e.message}")
