#!/usr/bin/env python3
import argparse, os, re, json, glob, importlib

# NOTE: torch / transformers are imported lazily inside load_model / score_chunks
# so that the pure helpers (extract_functions, chunk_lines, load_threshold) stay
# importable without the heavy ML stack installed (e.g. in unit tests / CI).

from profiles import select_profile, add_profile_arg

C_EXTS = (".c", ".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx", ".h++",
          ".tcc", ".ipp", ".inl")

# Extensions parsed with the C++ grammar. ".h" is ambiguous (C or C++ header)
# and handled separately, preferring C++ since that grammar is a near-superset
# and still parses plain C definitions correctly.
_CPP_EXTS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++",
             ".tcc", ".ipp", ".inl"}


def list_sources(src):
    out = []
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            for fn in files:
                if fn.lower().endswith(C_EXTS):
                    out.append(os.path.join(root, fn))
    else:
        out.append(src)
    return sorted(out)


def chunk_lines(lines, window=12, step=6):
    i = 0
    n = len(lines)
    while i < n:
        j = min(n, i + window)
        yield i, j, lines[i:j]
        if j == n:
            break
        i += step


_RAW_DELIM_BAD_RE = re.compile(r'[\s()\\"]')  # chars a raw-string delimiter can't hold


def mask_comments_and_strings(code):
    """Return `code` with comment text and string/char-literal contents replaced
    by spaces. The result has IDENTICAL length and line structure: delimiters
    (//, /*, */, quotes) and every newline are kept, so line/column positions
    map 1:1 onto the original. Handles line comments, block comments spanning
    lines, string and char literals, escaped quotes, C++11 raw string literals
    R"delim(...)delim" (masked ACROSS newlines — no escape processing applies
    inside them), and C++14 digit separators (the apostrophe in 10'000 is not a
    char-literal opener). An unterminated ordinary string literal ends at the
    line break (like the C lexer), so a stray quote cannot poison the rest of
    the file."""
    out = []
    i, n = 0, len(code)
    state = None  # None | "line" | "block" | '"' | "'"
    while i < n:
        c = code[i]
        if c == "\n":
            if state in ('"', "'", "line"):
                state = None
            out.append(c)
            i += 1
        elif state == "line":
            out.append(" ")
            i += 1
        elif state == "block":
            if code.startswith("*/", i):
                out.append("*/")
                state = None
                i += 2
            else:
                out.append(" ")
                i += 1
        elif state in ('"', "'"):
            if c == "\\" and i + 1 < n:
                # escape pair: mask both chars, but keep a line continuation's
                # newline so the line structure survives.
                out.append(" \n" if code[i + 1] == "\n" else "  ")
                i += 2
            elif c == state:
                out.append(c)
                state = None
                i += 1
            else:
                out.append(" ")
                i += 1
        else:
            if code.startswith("//", i):
                out.append("//")
                state = "line"
                i += 2
            elif code.startswith("/*", i):
                out.append("/*")
                state = "block"
                i += 2
            elif (code.startswith('R"', i)
                  and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_")
                       or code[i - 1] in "LuU8")):
                # C++11 raw string literal R"delim( ... )delim" (optionally
                # behind an L/u/U/u8 encoding prefix). No escape processing
                # applies and the literal may span lines, so its contents
                # (and the delimiter, which may itself hold braces) are
                # masked to spaces ACROSS newlines until ')delim"'; newlines
                # are kept so the line structure survives.
                lp = code.find("(", i + 2)
                delim = code[i + 2:lp] if lp != -1 else None
                if (delim is not None and len(delim) <= 16
                        and not _RAW_DELIM_BAD_RE.search(delim)):
                    close = code.find(")" + delim + '"', lp + 1)
                    # index just past the closing '"'; unterminated -> EOF
                    end = close + len(delim) + 2 if close != -1 else n
                    body_end = end - 1 if close != -1 else n
                    out.append('R"')
                    for ch in code[i + 2:body_end]:
                        out.append(ch if ch == "\n" else " ")
                    if close != -1:
                        out.append('"')
                    i = end
                else:
                    # malformed delimiter: not a raw string. 'R' is plain
                    # code; the quote is handled on the next iteration.
                    out.append(c)
                    i += 1
            else:
                if c == "'" and i > 0 and (code[i - 1].isalnum()
                                           or code[i - 1] == "_"):
                    # C++14 digit separator (10'000, 0xBEEF'FACE): an
                    # apostrophe glued to an identifier/number char is not a
                    # char-literal opener — unless that char ends an
                    # encoding prefix (L'x', u8'\0').
                    k = i
                    while k > 0 and (code[k - 1].isalnum() or code[k - 1] == "_"):
                        k -= 1
                    if code[k:i] not in ("L", "u", "U", "u8"):
                        out.append(c)
                        i += 1
                        continue
                if c in "\"'":
                    state = c
                out.append(c)
                i += 1
    return "".join(out)


_ACCESS_SPEC_RE = re.compile(r"^(public|protected|private)\s*(slots|signals)?\s*:$")


def _signature_start(clean, brace_idx):
    """Given comment/string-stripped lines and the index of the line holding a
    depth-0 opening '{', walk back to the first line of the signature.

    The signature begins right after the previous statement/block boundary: a
    line whose (stripped) content ends with ';', '}', '{', or is blank, a
    preprocessor directive, or a class access specifier (public:/private:/
    protected:). This handles multi-line signatures, K&R param declarations,
    structs, and same-line braces without the old fixed -2 guess.
    """
    start = brace_idx
    while start > 0:
        prev = clean[start - 1].strip()
        if (prev == "" or prev.endswith(";") or prev.endswith("{")
                or prev.endswith("}") or prev.startswith("#")
                or _ACCESS_SPEC_RE.match(prev)):
            break
        start -= 1
    return start


def _window_long_spans(spans, lines, max_lines):
    """Shared post-processing: emit (start, end, lines) tuples, windowing any
    span longer than max_lines so nothing blows past the tokenizer limit."""
    out = []
    for a, b in spans:
        b = min(b, len(lines))
        if b <= a:
            continue
        if b - a <= max_lines:
            out.append((a, b, lines[a:b]))
        else:  # huge function: window it so nothing exceeds the token limit badly
            for i0, i1, chunk in chunk_lines(lines[a:b], window=60, step=30):
                out.append((a + i0, a + i1, chunk))
    return out


# --- optional tree-sitter backend -------------------------------------------
# tree-sitter gives a real parse tree, so it nails function boundaries on
# macro-heavy, K&R, and deeply nested code where the brace heuristic guesses,
# and it understands C++ (class methods, constructors, templates, namespaces,
# operator overloads) which the brace matcher cannot. It is an OPTIONAL
# dependency: when absent we transparently fall back to the brace matcher,
# keeping the core install light and CI simple.
_TS_MODULES = {"c": "tree_sitter_c", "cpp": "tree_sitter_cpp"}
_TS_PARSERS = {}  # normalized lang -> Parser or None (cached, including failures)


def _normalize_lang(lang):
    return "cpp" if str(lang or "c").lower() in ("cpp", "c++", "cxx", "cc") else "c"


def _treesitter_parser(lang="c"):
    """Return a cached tree-sitter Parser for `lang` ("c"|"cpp"), or None."""
    lang = _normalize_lang(lang)
    if lang in _TS_PARSERS:
        return _TS_PARSERS[lang]
    parser = None
    try:
        grammar = importlib.import_module(_TS_MODULES[lang])
        from tree_sitter import Language, Parser

        language = Language(grammar.language())
        try:
            parser = Parser(language)  # tree_sitter >= 0.22
        except TypeError:  # pragma: no cover - older tree_sitter API
            parser = Parser()
            if hasattr(parser, "set_language"):
                parser.set_language(language)
            else:
                parser.language = language
    except Exception:  # ImportError, ABI mismatch, ... -> brace fallback
        parser = None
    _TS_PARSERS[lang] = parser
    return parser


def treesitter_available(lang="c"):
    """True when the tree-sitter backend for `lang` ("c"|"cpp") can be loaded."""
    return _treesitter_parser(lang) is not None


def lang_for_path(path, default="c"):
    """Pick the tree-sitter grammar ("c"|"cpp") for a source path by extension.

    C++ extensions -> "cpp"; ".c" -> "c"; ".h" headers prefer "cpp" when that
    grammar is installed (it parses C too); anything else -> `default`.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _CPP_EXTS:
        return "cpp"
    if ext == ".h":
        return "cpp" if treesitter_available("cpp") else "c"
    if ext == ".c":
        return "c"
    return default


def _template_wraps_function(node):
    """True when a template_declaration wraps a function_definition, directly
    or through nested template_declarations (out-of-line member templates carry
    doubled `template<..> template<..>` headers)."""
    for c in node.children:
        if c.type == "function_definition":
            return True
        if c.type == "template_declaration" and _template_wraps_function(c):
            return True
    return False


def _declares_lambda(node):
    """True when a declaration node holds a lambda_expression initializer
    (e.g. `auto handler = [](int x) { ... };` at namespace/file scope)."""
    stack = list(node.children)
    while stack:
        c = stack.pop()
        if c.type == "lambda_expression":
            return True
        stack.extend(c.children)
    return False


def _extract_functions_treesitter(code, max_lines=200, lang="c"):
    """Extract whole functions/methods using a tree-sitter parse tree.

    Handles C and C++: free functions, class member functions, constructors,
    destructors, operator overloads, out-of-line definitions, function
    templates (the enclosing `template<...>` header is kept — including doubled
    headers on out-of-line member templates), and namespace/file-scope lambdas
    assigned to variables. Returns the same (start_idx, end_idx, lines) shape
    as the brace matcher (0-based [start,end)), None when the file declares no
    functions, or None if tree-sitter is absent.
    """
    parser = _treesitter_parser(lang)
    if parser is None:
        return None
    lines = code.splitlines()
    tree = parser.parse(code.encode("utf-8", errors="ignore"))
    spans = []
    # Iterative pre-order walk. We do NOT descend into a captured function (its
    # body — incl. lambdas — belongs to that one span), but we DO descend into
    # classes/namespaces/class-templates to capture each member function.
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        t = node.type
        if t == "function_definition":
            spans.append((node.start_point[0], node.end_point[0] + 1))
            continue
        if t == "template_declaration" and _template_wraps_function(node):
            # function template: capture the whole thing so the template<...>
            # header(s) travel with the function body.
            spans.append((node.start_point[0], node.end_point[0] + 1))
            continue
        if t == "declaration" and _declares_lambda(node):
            # namespace/file-scope lambda: only reachable here when OUTSIDE any
            # function body (we never descend into captured functions).
            spans.append((node.start_point[0], node.end_point[0] + 1))
            continue
        stack.extend(reversed(node.children))
    if not spans:
        return None
    spans.sort()
    return _window_long_spans(spans, lines, max_lines)


def extract_functions(code, max_lines=200, parser=None, lang="c"):
    """Extract whole functions from C/C++ source for whole-function scoring.

    parser selects the boundary engine:
      "auto" (default) - tree-sitter when installed, else the brace matcher.
      "treesitter"     - force tree-sitter (raises if it cannot be loaded).
      "brace"          - force the dependency-free brace heuristic.
    When parser is None the CCR_PARSER env var is consulted, defaulting to auto.
    lang ("c"|"cpp") picks the tree-sitter grammar; the brace fallback is
    language-agnostic.

    Returns (start_idx, end_idx, lines) tuples, 0-based [start, end), or None
    when the file yields no functions (headers, macros).
    """
    mode = (parser or os.environ.get("CCR_PARSER") or "auto").strip().lower()
    if mode in ("ts", "tree-sitter", "tree_sitter"):
        mode = "treesitter"
    if mode == "treesitter":
        if not treesitter_available(lang):
            grammar = _TS_MODULES[_normalize_lang(lang)]
            raise RuntimeError(
                f"parser='treesitter' requested but the tree-sitter {lang} grammar "
                f"is not installed. Install it with: pip install tree_sitter {grammar}"
            )
        return _extract_functions_treesitter(code, max_lines, lang)
    if mode == "auto":
        res = _extract_functions_treesitter(code, max_lines, lang)
        if res is not None:
            return res
        # tree-sitter unavailable, or parsed but found nothing -> brace matcher.
    return _extract_functions_brace(code, max_lines)


# A '{' whose heading (signature start .. the brace) fully matches this opens a
# container block (class/struct/union/namespace/enum/extern "C"): its body is
# scanned for functions instead of being swallowed as one (an enum body simply
# yields nothing). Storage/qualifier prefixes are allowed so 'typedef struct {',
# 'static struct {', 'enum class X : int {' etc. are recognized as data blocks,
# not functions. `[^(){};=]*` allows the tag name and a base-clause but rejects
# parens ('struct point mk(int x) {' is a function returning struct) and '='
# ('struct cmd table[] = {' is data). String contents are masked, so
# extern "C" appears as extern " ".
_CONTAINER_RE = re.compile(
    r'^\s*(?:template\s*<[^>]*>\s*)?'
    r'(?:(?:typedef|static|const|constexpr|inline)\s+)*'
    r'(?:(?:class|struct|union|namespace|enum(?:\s+(?:class|struct))?)\b[^(){};=]*'
    r'|extern\s*"[^"]*"\s*)$'
)

# A K&R parameter declaration line ('int a;', 'char *s;', 'struct foo p[2];'):
# ends with ';' and holds no parens/braces/'='/strings. Used to let the
# function-detection look-back reach a signature's ')' across K&R param decls.
_KR_PARAM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\s,*\[\]]*;$")


def _extract_functions_brace(code, max_lines=200):
    """Best-effort C/C++ function extraction via brace matching.

    Dependency-free fallback used when tree-sitter is unavailable. The
    classifier is trained on whole functions (BigVul granularity), so scoring
    whole functions at inference avoids a train/test mismatch.

    Container-aware: a '{' opening a class/struct/union/namespace/enum/
    extern "C" block is transparent, so in-class methods, namespaced and
    extern-C functions are captured individually (signature..closing brace) at
    any container nesting depth. Enum/struct/array data definitions (incl.
    typedef/static/const-prefixed ones) and C++ brace initialization
    (counter{0}, ctor init-list x_{0}) yield no span.

    Returns (start_idx, end_idx, lines) tuples, 0-based [start, end).
    Functions longer than max_lines are split into sliding windows.
    Falls back to None when the file yields no functions (headers, macros).
    """
    lines = code.splitlines()
    # Masked copy: strings/comments neutralized, columns preserved. Comment
    # delimiters are blanked too so a comment-only line looks blank to
    # _signature_start (any surviving //, /*, */ is a real comment delimiter).
    clean = [ln.replace("//", "  ").replace("/*", "  ").replace("*/", "  ")
             for ln in mask_comments_and_strings(code).splitlines()]
    spans = []
    stack = []   # kind of every open brace: container | function | other | nested
    opaque = 0   # open braces that are NOT containers (function/other/nested)
    fn_start = None

    for idx, ln in enumerate(clean):
        for col, ch in enumerate(ln):
            if ch == "{":
                kind = "nested"
                if opaque == 0:  # enclosed only by containers (or nothing)
                    head_start = _signature_start(clean, idx)
                    heading = " ".join(clean[head_start:idx] + [ln[:col]])
                    if _CONTAINER_RE.match(heading):
                        kind = "container"
                    elif col > 0 and (ln[col - 1].isalnum() or ln[col - 1] == "_"):
                        # '{' glued to an identifier char is C++ brace
                        # initialization (counter{0}, int limit{100}, ctor
                        # init-list x_{0}), never a function body: a real
                        # definition keeps ')' (or, K&R, nothing) before it.
                        kind = "other"
                    else:
                        # heuristic: a ')' within the heading = function
                        # definition. The heading is bounded at the last
                        # statement boundary (_signature_start), so a
                        # PREVIOUS statement's ')' can never qualify this
                        # '{'. Only text BEFORE the brace counts
                        # ('= { (int)F(1),' initializers are not signatures)
                        # and an '=' after the last ')' means a braced
                        # initializer ('int x[N(a)] = {'), not a body.
                        look = heading
                        if ")" not in look:
                            # K&R: param decls (plus blank/preprocessor
                            # lines) may sit between the signature's ')'
                            # and the '{'.
                            k = head_start
                            while k > 0 and head_start - k < 40:
                                prev = clean[k - 1].strip()
                                if (prev == "" or prev.startswith("#")
                                        or _KR_PARAM_RE.match(prev)):
                                    k -= 1
                                else:
                                    break
                            if (k < head_start and k > 0
                                    and clean[k - 1].rstrip().endswith(")")):
                                look = " ".join(clean[k - 1:idx] + [ln[:col]])
                        if ")" in look and "=" not in look[look.rfind(")") + 1:]:
                            kind = "function"
                            fn_start = head_start
                        else:
                            kind = "other"
                stack.append(kind)
                if kind != "container":
                    opaque += 1
            elif ch == "}":
                if not stack:
                    continue
                kind = stack.pop()
                if kind != "container":
                    opaque -= 1
                if kind == "function":
                    spans.append((fn_start, idx + 1))
                    fn_start = None

    if not spans:
        return None
    return _window_long_spans(spans, lines, max_lines)


def load_inference_config(model_dir):
    """Read <model_dir>/inference.json (written by train_vuln_model.py).

    Always returns a dict with at least {"threshold": float|None,
    "max_length": int}; a missing file, unreadable JSON, or bad values fall
    back to {"threshold": None, "max_length": 512}. Extra keys the trainer
    persists (notes, provenance, ...) are passed through untouched."""
    cfg = {"threshold": None, "max_length": 512}
    path = os.path.join(model_dir, "inference.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return cfg
    if not isinstance(data, dict):
        return cfg
    cfg.update({k: v for k, v in data.items()
                if k not in ("threshold", "max_length")})
    try:
        cfg["threshold"] = float(data["threshold"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        max_length = int(data["max_length"])
        if max_length > 0:
            cfg["max_length"] = max_length
    except (KeyError, TypeError, ValueError):
        pass
    return cfg


def load_threshold(model_dir, default=0.5):
    """Threshold is written by train_vuln_model.py to inference.json.
    threshold.txt is honored for backward compatibility with older bundles."""
    thr = load_inference_config(model_dir)["threshold"]
    if thr is not None:
        return thr
    txt = os.path.join(model_dir, "threshold.txt")
    if os.path.exists(txt):
        try:
            return float(open(txt).read().strip())
        except ValueError:
            pass
    return default


def _assert_safetensors(model_dir):
    """Refuse to load legacy pickle weights: a malicious *.bin can execute
    arbitrary code on unpickle. Require a *.safetensors and reject a model
    directory that ships only pytorch_model.bin (no safetensors)."""
    has_safetensors = bool(glob.glob(os.path.join(model_dir, "*.safetensors")))
    has_pickle = bool(glob.glob(os.path.join(model_dir, "*.bin")) or
                      glob.glob(os.path.join(model_dir, "pytorch_model*.bin")))
    if not has_safetensors:
        if has_pickle:
            raise RuntimeError(
                f"Refusing to load {model_dir}: only legacy pickle weights "
                "(*.bin) found and no *.safetensors. Pickle weights can execute "
                "arbitrary code on load. Re-save the model with "
                "safe_serialization=True (produces *.safetensors)."
            )
        raise RuntimeError(
            f"No *.safetensors weights found in {model_dir}; cannot load model."
        )


def load_model(model_dir, dtype_name="float32"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = os.path.abspath(model_dir)
    _assert_safetensors(model_dir)
    revision = os.environ.get("HF_REVISION") or None
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True,
                                        revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True, revision=revision)
    thr = load_threshold(model_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and dtype_name in ("float16", "bfloat16"):
        model = model.to(dtype=getattr(torch, dtype_name))
    model.to(dev)
    model.eval()
    return tok, model, thr, dev


def score_chunks(tok, model, dev, code, batch_size=32, granularity="function",
                 parser=None, lang="c", max_length=512):
    """Score chunks of a file in batches. Default granularity is whole
    functions (matches BigVul training data); falls back to 12-line sliding
    windows for files where no functions are found. `parser` selects the
    function-boundary engine and `lang` ("c"|"cpp") the tree-sitter grammar
    (see extract_functions). `max_length` is the tokenizer truncation length
    and should match the model's training config (inference.json)."""
    import torch

    chunks = None
    if granularity == "function":
        chunks = extract_functions(code, parser=parser, lang=lang)
    if chunks is None:
        chunks = list(chunk_lines(code.splitlines()))
    results = []
    with torch.no_grad():
        for b in range(0, len(chunks), batch_size):
            batch = chunks[b:b + batch_size]
            texts = ["\n".join(c[2]) for c in batch]
            enc = tok(texts, truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            for (i0, i1, chunk), prob in zip(batch, probs):
                results.append({
                    "start_line": i0 + 1, "end_line": i1,
                    "score": float(prob), "snippet": "\n".join(chunk),
                })
    return results


def merge_overlapping(findings):
    """Merge overlapping/adjacent windows within a file into one region,
    keeping the max score. Windows overlap by design (step < window), so
    without this every hot spot is reported ~2x."""
    merged = []
    for f in sorted(findings, key=lambda x: x["start_line"]):
        if merged and f["start_line"] <= merged[-1]["end_line"] + 1:
            prev = merged[-1]
            if f["score"] > prev["score"]:
                prev["score"] = f["score"]
            if f["end_line"] > prev["end_line"]:
                # extend snippet with the non-overlapping tail lines
                new_lines = f["snippet"].splitlines()[prev["end_line"] - f["start_line"] + 1:]
                prev["snippet"] = prev["snippet"] + "\n" + "\n".join(new_lines) if new_lines else prev["snippet"]
                prev["end_line"] = f["end_line"]
        else:
            merged.append(dict(f))
    return merged


def _parser_engine(parser_mode, lang):
    """Boundary engine actually in play, for the startup banner. tree-sitter
    counts as available when the grammar for the requested lang loads — with
    lang=auto, when ANY grammar loads — not just the default C grammar."""
    if parser_mode not in ("auto", "treesitter"):
        return "brace"
    if lang in ("c", "cpp"):
        return "tree-sitter" if treesitter_available(lang) else "brace"
    if treesitter_available("c") or treesitter_available("cpp"):
        return "tree-sitter"
    return "brace"


def main():
    ap = argparse.ArgumentParser(description="Scan C/C++ sources with the vuln classifier")
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default="scan_out")
    ap.add_argument("--model", default="vuln-model")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the model's tuned threshold from inference.json")
    ap.add_argument("--max-length", type=int, default=None,
                    help="Tokenizer truncation length (default: max_length from "
                         "the model's inference.json, else 512)")
    ap.add_argument("--max-per-file", type=int, default=0,
                    help="Cap findings per file after merging (0 = no cap)")
    ap.add_argument("--granularity", choices=["function", "window"], default="function",
                    help="Score whole functions (matches training data) or 12-line windows")
    ap.add_argument("--parser", choices=["auto", "treesitter", "brace"], default="auto",
                    help="Function-boundary engine: auto (tree-sitter if installed), "
                         "treesitter (force), or brace (dependency-free fallback)")
    ap.add_argument("--lang", choices=["auto", "c", "cpp"], default="auto",
                    help="tree-sitter grammar: auto (per file extension), or force c/cpp")
    add_profile_arg(ap)
    args = ap.parse_args()

    prof_name, prof = select_profile(args.profile)
    parser_engine = _parser_engine(args.parser, args.lang)
    print(f"[profile] {prof_name}: batch={prof['classifier_batch_size']} "
          f"dtype={prof['classifier_dtype']} parser={parser_engine} lang={args.lang}")

    os.makedirs(args.out, exist_ok=True)
    tok, model, thr, dev = load_model(args.model, prof["classifier_dtype"])
    if args.threshold is not None:
        thr = args.threshold
    max_length = (args.max_length if args.max_length
                  else load_inference_config(args.model)["max_length"])
    print(f"[model] device={dev} threshold={thr:.2f} max_length={max_length}")

    all_findings = []
    for path in list_sources(args.src):
        with open(path, "r", errors="ignore", encoding="utf-8") as fh:
            code = fh.read()
        lang = args.lang if args.lang in ("c", "cpp") else lang_for_path(path)
        chunks = score_chunks(tok, model, dev, code, prof["classifier_batch_size"],
                              granularity=args.granularity, parser=args.parser, lang=lang,
                              max_length=max_length)
        hot = [c for c in chunks if c["score"] >= thr]
        hot = merge_overlapping(hot)
        hot.sort(key=lambda x: x["score"], reverse=True)
        if args.max_per_file > 0:
            hot = hot[:args.max_per_file]
        all_findings.extend({"file": path, **c} for c in hot)

    out_json = os.path.join(args.out, "classifier_findings.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "profile": prof_name, "findings": all_findings}, f, indent=2)
    print(f"[OK] {len(all_findings)} findings -> {out_json}")

    topn = sorted(all_findings, key=lambda x: x["score"], reverse=True)[:4]
    if topn:
        print("\nTop findings:")
        for t in topn:
            print(f"  {t['file']}:{t['start_line']}-{t['end_line']}  score={t['score']:.2f}")


if __name__ == "__main__":
    main()
