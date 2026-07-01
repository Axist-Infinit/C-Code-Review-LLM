#!/usr/bin/env python3
import argparse, functools, os, json, re, html

# --------------------------------------------------------------------
# Comment/string masking. Patterns are matched against masked text so that
# commented-out code (`/* strcpy(d, s); */`) and log strings ("call system()")
# cannot trip API regexes. Masking is same-shape (char-for-char, newlines
# kept), so match offsets and line numbers align with the original snippet.
# Patterns that must inspect string literal contents (scanf format strings,
# hardcoded credentials) set "needs_strings": True and are matched against a
# variant where only comments are masked. Patterns whose TRIGGER must not see
# literals but whose confirmation must (the '\0' terminator store after
# strncpy) set "confirm_needs_strings": True instead.
# --------------------------------------------------------------------

# C++11 raw string literals: R"delim( ... )delim", optionally u8/u/U/L-prefixed.
# Backslashes and quotes inside the body are literal, so the lexer must consume
# the whole literal as one unit or it desyncs and masks real code after it.
_RAW_DELIM_RE = re.compile(r'([^\s()\\"]{0,16})\(')


def _raw_string_open(code, i):
    """True when code[i] is the '"' of a raw string opener (R", u8R", LR", ...)."""
    j = i - 1
    if j < 0 or code[j] != "R":
        return False
    p = j - 1
    if p >= 1 and code[p] == "8" and code[p - 1] == "u":
        p -= 2
    elif p >= 0 and code[p] in "uUL":
        p -= 1
    # the prefix must be a token of its own, not the tail of an identifier
    return not (p >= 0 and (code[p].isalnum() or code[p] == "_"))


def _raw_string_span(code, i):
    """For the opening '"' of a raw string at code[i], return (content_start,
    content_end, literal_end): the body between the parens and the index just
    past the closing quote. None when malformed/unterminated — the caller then
    falls back to ordinary string handling."""
    m = _RAW_DELIM_RE.match(code, i + 1)
    if m is None:
        return None
    closer = ")" + m.group(1) + '"'
    k = code.find(closer, m.end())
    if k == -1:
        return None
    return m.end(), k, k + len(closer)


def _mask_code(code, mask_strings=True):
    """Return `code` with comments (always) and string/char literal contents
    (when mask_strings) replaced by spaces. Same length, newlines preserved.
    Raw string literals (R"delim(...)delim") are lexed as single units so
    embedded quotes/braces cannot desync the masker."""
    out = list(code)
    i, n = 0, len(code)
    state = None  # None (code) | "line" | "block" | '"' | "'"
    while i < n:
        c = code[i]
        if state is None:
            if c == "/" and i + 1 < n and code[i + 1] == "/":
                state = "line"
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if c == "/" and i + 1 < n and code[i + 1] == "*":
                state = "block"
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if c == '"' and _raw_string_open(code, i):
                span = _raw_string_span(code, i)
                if span is not None:
                    cs, ce, le = span
                    if mask_strings:
                        for k in range(cs, ce):
                            if code[k] != "\n":
                                out[k] = " "
                    i = le
                    continue
                # malformed/unterminated raw literal: ordinary handling below
            # A ' preceded by a word char is a C++14 digit separator (1'000),
            # not a char literal — treating it as one would mask real code.
            if c == '"' or (c == "'" and not (i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"))):
                state = c
                i += 1
                continue
            i += 1
            continue
        if state == "line":
            if c == "\n":
                state = None
            else:
                out[i] = " "
            i += 1
            continue
        if state == "block":
            if c == "*" and i + 1 < n and code[i + 1] == "/":
                out[i] = out[i + 1] = " "
                state = None
                i += 2
                continue
            if c != "\n":
                out[i] = " "
            i += 1
            continue
        # inside a string/char literal
        if c == "\\" and i + 1 < n:
            if mask_strings:
                out[i] = " "
                if code[i + 1] != "\n":
                    out[i + 1] = " "
            i += 2
            continue
        if c == state:
            state = None
            i += 1
            continue
        if c == "\n":
            # unterminated literal: bail back to code state instead of masking
            # the rest of the snippet (defensive on macro/snippet fragments)
            state = None
            i += 1
            continue
        if mask_strings:
            out[i] = " "
        i += 1
    return "".join(out)


def _mask_fallback(code):
    """Local stand-in for local_vuln_scanner.mask_comments_and_strings."""
    return _mask_code(code, mask_strings=True)


_mask_all_impl = None


def _mask_all(code):
    """Comments AND string contents masked (shared helper when available)."""
    global _mask_all_impl
    if _mask_all_impl is None:
        try:
            from local_vuln_scanner import mask_comments_and_strings as impl
        except Exception:  # helper not present in this checkout
            impl = _mask_fallback
        _mask_all_impl = impl
    return _mask_all_impl(code)


# --------------------------------------------------------------------
# Inline suppression, mirroring heuristic_scan's semantics so re-explaining a
# snippet downstream (explain_findings CLI, llm_explain heuristic backend)
# cannot resurrect a rule the scan already suppressed: a match whose line
# contains `ccr-ignore` is dropped; `ccr-ignore: rule1,rule2` limits the
# suppression to those pattern ids. Checked against the RAW snippet lines
# because the marker lives in a comment, which masking blanks out.
# --------------------------------------------------------------------

_IGNORE_RE = re.compile(r"ccr-ignore(?:\s*:\s*([\w\s,-]+))?")


def _suppressed_rules(line):
    """None when the line has no ccr-ignore; empty set = every rule suppressed."""
    m = _IGNORE_RE.search(line)
    if m is None:
        return None
    if m.group(1):
        return {r.strip() for r in m.group(1).split(",") if r.strip()}
    return set()


# --------------------------------------------------------------------
# Context-sensitive confirmation helpers. A pattern may carry an optional
# "confirm": callable(text, match) -> bool; a regex hit only counts when the
# callable returns True, letting a rule look before/after its trigger (e.g.
# free(p) ... *p) without a monster regex. `text` is the masked snippet the
# trigger matched in.
# --------------------------------------------------------------------

_ASSIGN_NEXT = re.compile(r"\s*=(?!=)")
_DEREF_NEXT = re.compile(r"\s*(?:->|\[)")
_REINIT_NEXT = re.compile(r"\s*\.\s*(?:clear|reset|assign)\s*\(")


def _call_args(text, open_idx):
    """Top-level comma-split argument strings of the call whose '(' is at
    text[open_idx]; None when the parens never balance (truncated snippet)."""
    if open_idx >= len(text) or text[open_idx] != "(":
        return None
    depth, args, start = 0, [], open_idx + 1
    for i in range(open_idx, len(text)):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append(text[start:i])
                return args
        elif c == "," and depth == 1:
            args.append(text[start:i])
            start = i + 1
    return None


_SIZEOF_ARG_RE = re.compile(
    r"^sizeof(?:\s*\(\s*(?P<par>.+)\s*\)|\s+(?P<bare>[A-Za-z_][\w.\[\]>-]*))\s*$")


def _mem_len_not_dst_sizeof(text, m):
    """Keep a generic memcpy/memmove hit unless the length argument is exactly
    sizeof(<destination arg>) — a copy bounded by the destination's own size."""
    args = _call_args(text, m.end() - 1)
    if not args or len(args) < 3:
        return True
    mo = _SIZEOF_ARG_RE.match(args[2].strip())
    if not mo:
        return True
    operand = re.sub(r"\s+", "", mo.group("par") or mo.group("bare") or "")
    return operand != re.sub(r"\s+", "", args[0])


def _confirm_new_delete_mismatch(text, m):
    array_delete, ident = m.group(1), re.escape(m.group(2))
    if array_delete:  # delete[] p — mismatch when p came from scalar new
        return re.search(
            r"\b%s\s*=\s*new\s+[\w:]+(?:\s*<[^;\n]*>)?\s*(?:\(|\{|;)" % ident, text) is not None
    return re.search(  # delete p — mismatch when p came from new T[...]
        r"\b%s\s*=\s*new\s+[\w:]+(?:\s*<[^;\n]*>)?\s*\[" % ident, text) is not None


def _confirm_use_after_move(text, m):
    ident = m.group(1)
    nxt = re.compile(r"\b%s\b" % re.escape(ident)).search(text, m.end())
    if not nxt:
        return False
    # `x = ...` / `x.clear()` right after the move re-initialises the object
    if _ASSIGN_NEXT.match(text, nxt.end()) or _REINIT_NEXT.match(text, nxt.end()):
        return False
    return True


def _confirm_dangling_cstr(text, m):
    ret_id = m.group(1)
    if ret_id is None:  # `char *p = s.c_str()` — pointer aliases string internals
        return True
    # `return t.c_str()` only counts when t is a std::string declared locally
    decl = re.compile(r"\b(?:std\s*::\s*)?string\s+[\w\s,*&]*\b%s\b" % re.escape(ret_id))
    return decl.search(text, 0, m.start()) is not None


def _confirm_strncpy_no_nul(text, m):
    dst = re.escape(m.group(1))
    return re.search(r"%s\s*\[[^\]\n]*\]\s*=\s*(?:'\\0'|0\b)" % dst, text) is None


def _confirm_free_then_use(text, m):
    ident = m.group(1)
    for um in re.compile(r"\b%s\b" % re.escape(ident)).finditer(text, m.end()):
        s, e = um.start(), um.end()
        if _ASSIGN_NEXT.match(text, e):  # p = ... — reassigned before any use
            return False
        if _DEREF_NEXT.match(text, e):   # p-> or p[
            return True
        k = s - 1
        while k >= 0 and text[k] in " \t":
            k -= 1
        if k >= 0 and text[k] == "*":    # *p
            return True
    return False


def _confirm_double_free(text, m):
    ident = re.escape(m.group(1))
    nxt = re.compile(r"\bfree\s*\(\s*%s\s*\)" % ident).search(text, m.end())
    if not nxt:
        return False
    assign = re.compile(r"\b%s\s*=(?!=)" % ident).search(text, m.end())
    return assign is None or assign.start() > nxt.start()


def _confirm_iterator_invalidation(text, m):
    cont = m.group(1)
    n = len(text)
    i = m.end()
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i < n and text[i] == "{":  # braced body: match to the closing brace
        depth, j = 0, i
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[i:j + 1]
    else:  # single-statement body
        j = text.find(";", i)
        body = text[i:(j + 1 if j != -1 else n)]
    return re.search(
        r"\b%s\s*\.\s*(?:push_back|insert|erase|emplace_back|emplace)\s*\(" % re.escape(cont),
        body) is not None


# A checked pointer is often a member chain (s->buf, cfg.ptr): match the whole
# chain and credit every component identifier, so `if (!s->buf)` counts as a
# null check for the `s->buf = malloc(...)` trigger (which captures the member
# name `buf`, not the base object).
_ID_CHAIN = r"[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*)*"


@functools.lru_cache(maxsize=8)
def _null_checked_idents(text):
    """Identifiers appearing in any NULL-check context in `text`. Computed once
    per snippet (all malloc triggers share the same masked text) so the confirm
    below stays linear on files with many allocations."""
    ids = set()
    for pat in (r"!\s*(%s)\b" % _ID_CHAIN,
                r"\b(%s)\s*[=!]=\s*(?:NULL|nullptr|0)\b" % _ID_CHAIN,
                r"\b(?:NULL|nullptr|0)\s*[=!]=\s*(%s)\b" % _ID_CHAIN,
                r"\b(?:if|while|assert)\s*\(\s*(%s)\s*(?:\)|&&|\|\|)" % _ID_CHAIN):
        for mm in re.finditer(pat, text):
            ids.update(re.findall(r"[A-Za-z_]\w*", mm.group(1)))
    return frozenset(ids)


def _confirm_malloc_no_null_check(text, m):
    esc = re.escape(m.group(1))
    if m.group(1) in _null_checked_idents(text):
        return False  # a null check exists somewhere — stay quiet
    end = m.end()
    for _ in range(4):  # rest of the alloc line + the next ~3 lines
        nl = text.find("\n", end)
        if nl == -1:
            end = len(text)
            break
        end = nl + 1
    window = text[m.end():end]
    # sizeof(*p) inspects the type, not the pointer — drop before deref checks
    window = re.sub(r"\bsizeof\s*\([^()\n]*\)|\bsizeof\s+\**\s*\w+", " ", window)
    for pat in (r"\b%s\s*->" % esc, r"\*\s*%s\b" % esc, r"\b%s\s*\[" % esc,
                r"\bmem(?:cpy|move|set)\s*\(\s*%s\s*," % esc,
                r"\bstr\w*cpy\s*\(\s*%s\s*," % esc):
        if re.search(pat, window):
            return True
    return False


# --------------------------------------------------------------------
# Pattern definitions: each ties a concrete API pattern to a specific
# explanation + recommendations.
# --------------------------------------------------------------------

# Each pattern carries a CWE id so heuristic findings get a meaningful SARIF rule
# id (not just the API name). Order is priority order: cwe_for_patterns() returns
# the first matched pattern's CWE, so keep the most specific/severe items first.
#
# Every pattern also carries three human-facing description fields that drive the
# structured review output the pipeline now emits:
#   "what" -> "What the code is doing:"      (generic description of the construct)
#   "risk" -> "What could go wrong:"          (the security failure mode)
#   "vuln" -> the vulnerability that was identified (its class/name)
# "why" stays as the one-line summary used for backward-compatible explanations.
#
# Optional per-pattern keys:
#   "needs_strings": True -> matched against text where only comments are
#                            masked (the rule inspects string literal contents)
#   "confirm": callable(text, match) -> bool -> contextual second check; a
#                            regex hit only counts when it returns True
#   "confirm_needs_strings": True -> only the confirm callable receives the
#                            strings-kept variant (same shape, so the trigger
#                            match's offsets stay valid) — for confirmations
#                            that must see literal contents ('\0' stores)
PATTERNS = [
    {
        "id": "gets",
        "regex": re.compile(r"\bgets\s*\("),
        "cwe": "CWE-242",  # use of inherently dangerous function
        "severity": "critical",
        "why": "Use of gets() is unsafe and leads to buffer overflow.",
        "what": "Reads a line from standard input into a buffer using gets().",
        "risk": "gets() provides no way to bound the input length, so any input longer than the "
                "destination buffer overflows it — a trivially exploitable stack buffer overflow.",
        "vuln": "Stack buffer overflow via unbounded input (gets)",
        "recs": [
            "Replace gets() with fgets() and pass the destination buffer size.",
            "Ensure all input functions have explicit maximum lengths."
        ],
    },
    {
        "id": "system",
        "regex": re.compile(r"\bsystem\s*\("),
        "cwe": "CWE-78",  # OS command injection
        "severity": "high",
        "why": "Passing attacker-influenced data to system() invites OS command injection.",
        "what": "Executes a command string through the shell via system().",
        "risk": "If any part of the command string is influenced by untrusted input, an attacker can "
                "inject additional shell commands (e.g. appending `; rm -rf /`).",
        "vuln": "OS command injection",
        "recs": [
            "Avoid the shell: use execve()/posix_spawn() with an explicit argument vector.",
            "If a shell is unavoidable, strictly allow-list and escape every argument."
        ],
    },
    {
        "id": "popen",
        "regex": re.compile(r"\bpopen\s*\("),
        "cwe": "CWE-78",  # OS command injection
        "severity": "high",
        "why": "popen() runs its argument through /bin/sh; untrusted input enables command injection.",
        "what": "Spawns a process by running a command string through /bin/sh via popen().",
        "risk": "Untrusted data interpolated into the command string lets an attacker run arbitrary "
                "shell commands.",
        "vuln": "OS command injection",
        "recs": [
            "Replace popen() with a direct exec of the target binary and an argument vector.",
            "Never interpolate untrusted data into a popen() command string."
        ],
    },
    {
        "id": "strcpy",
        "regex": re.compile(r"\bstrcpy\s*\("),
        "cwe": "CWE-120",  # classic buffer overflow
        "severity": "high",
        "why": "strcpy() does not check destination size and can overflow the target buffer.",
        "what": "Copies a NUL-terminated source string into a destination buffer with strcpy().",
        "risk": "strcpy() copies until the source's NUL byte with no regard for the destination's "
                "capacity; a longer source overflows the destination buffer.",
        "vuln": "Buffer overflow (unbounded string copy)",
        "recs": [
            "Replace strcpy() with strncpy()/strlcpy() and pass the destination size.",
            "Validate that the source string length is less than the destination buffer."
        ],
    },
    {
        "id": "strcat",
        "regex": re.compile(r"\bstrcat\s*\("),
        "cwe": "CWE-120",
        "severity": "high",
        "why": "strcat() appends without bounds checks and can overflow the destination buffer.",
        "what": "Appends a source string onto the end of a destination buffer with strcat().",
        "risk": "strcat() performs no bounds checking; if the combined length exceeds the destination "
                "capacity it overflows the buffer.",
        "vuln": "Buffer overflow (unbounded concatenation)",
        "recs": [
            "Replace strcat() with strncat()/strlcat() and pass the remaining buffer capacity.",
            "Prefer constructing strings via snprintf() or std::string‑like abstractions where possible."
        ],
    },
    {
        "id": "sprintf",
        "regex": re.compile(r"\bsprintf\s*\("),
        "cwe": "CWE-120",
        "severity": "high",
        "why": "sprintf() can overflow fixed‑size buffers if the formatted output is too long.",
        "what": "Formats data into a fixed destination buffer with sprintf().",
        "risk": "sprintf() does not know the destination size; an over-long formatted result (often "
                "from attacker-controlled fields) overflows the buffer.",
        "vuln": "Buffer overflow (unbounded formatting)",
        "recs": [
            "Replace sprintf() with snprintf() and cap the output size to the destination buffer.",
            "Validate that attacker‑controlled input is length‑checked before formatting."
        ],
    },
    {
        "id": "scanf_percent_s",
        # FIXED: properly quoted "%s" so the regex is valid Python.
        "regex": re.compile(r'\bscanf\s*\(\s*"%s"'),
        "needs_strings": True,  # the trigger lives inside the format literal
        "cwe": "CWE-120",
        "severity": "high",
        "why": 'scanf("%s") with no width specifier can overflow destination buffers.',
        "what": 'Reads a whitespace-delimited token from input with scanf("%s").',
        "risk": 'Without a width specifier, scanf("%s") writes an arbitrarily long token into a '
                "fixed-size buffer, overflowing it.",
        "vuln": "Buffer overflow via unbounded scanf",
        "recs": [
            'Use scanf with a width limit (e.g. scanf("%15s", buf)) or use fgets() with an explicit buffer size.',
            "Avoid reading unbounded strings into fixed‑size buffers."
        ],
    },
    {
        "id": "memcpy",
        "regex": re.compile(r"\bmemcpy\s*\("),
        # length == sizeof(dst) is bounded by construction — don't flag it
        "confirm": _mem_len_not_dst_sizeof,
        "cwe": "CWE-120",
        "severity": "medium",
        "why": "memcpy() with unchecked length can overflow or over‑read buffers.",
        "what": "Copies a number of bytes between buffers with memcpy().",
        "risk": "If the length is attacker-influenced or not validated against both buffer sizes, "
                "memcpy() over-reads the source or overflows the destination.",
        "vuln": "Buffer overflow / over-read (unchecked length)",
        "recs": [
            "Ensure the length passed to memcpy() is validated against both source and destination buffer sizes.",
            "Prefer higher‑level copy helpers where buffer sizes are tracked explicitly."
        ],
    },
    {
        "id": "memmove",
        "regex": re.compile(r"\bmemmove\s*\("),
        "confirm": _mem_len_not_dst_sizeof,
        "cwe": "CWE-120",
        "severity": "medium",
        "why": "memmove() with an unchecked length can overflow the destination buffer.",
        "what": "Copies a number of bytes between (possibly overlapping) buffers with memmove().",
        "risk": "An unchecked or attacker-influenced length overflows the destination buffer.",
        "vuln": "Buffer overflow (unchecked length)",
        "recs": [
            "Validate the length against both buffer sizes before calling memmove().",
            "Track buffer capacities explicitly (e.g. std::span) instead of raw pointers + length."
        ],
    },
    {
        "id": "alloca",
        "regex": re.compile(r"\balloca\s*\("),
        "cwe": "CWE-770",  # allocation without limits -> stack exhaustion
        "severity": "medium",
        "why": "alloca() allocates on the stack; an attacker-influenced size can exhaust the stack.",
        "what": "Allocates memory on the stack with alloca().",
        "risk": "A large or attacker-influenced size exhausts the stack, crashing the process or "
                "corrupting adjacent stack frames.",
        "vuln": "Stack exhaustion / uncontrolled allocation",
        "recs": [
            "Use a bounded stack buffer or heap allocation (with a checked size) instead of alloca().",
            "Cap and validate any size derived from input before allocating."
        ],
    },
    # --- C++-specific footguns ----------------------------------------------
    {
        "id": "reinterpret_cast",
        "regex": re.compile(r"\breinterpret_cast\s*<"),
        "cwe": "CWE-704",  # incorrect type conversion
        "severity": "medium",
        "why": "reinterpret_cast performs an unchecked reinterpretation; misuse causes type confusion and UB.",
        "what": "Reinterprets a pointer or value as an unrelated type with reinterpret_cast.",
        "risk": "The cast performs no checking; if the source and destination types are not "
                "layout-compatible the result is type confusion and undefined behavior.",
        "vuln": "Type confusion (unchecked reinterpretation)",
        "recs": [
            "Prefer static_cast/dynamic_cast, or std::bit_cast for value reinterpretation.",
            "Verify the source and destination types are layout-compatible before reinterpreting."
        ],
    },
    {
        "id": "const_cast",
        "regex": re.compile(r"\bconst_cast\s*<"),
        "cwe": "CWE-704",
        "severity": "low",
        "why": "const_cast strips constness; writing through it to a truly const object is undefined behavior.",
        "what": "Removes the const qualifier from a reference or pointer with const_cast.",
        "risk": "Writing through a const_cast to an object that is genuinely const is undefined "
                "behavior and can corrupt memory the compiler assumed immutable.",
        "vuln": "Undefined behavior via const removal",
        "recs": [
            "Redesign the interface so the object is mutable where it must be written.",
            "Never const_cast away const to modify an object that was originally declared const."
        ],
    },
    {
        "id": "printf_format_string",
        # printf called with a single bare identifier as the format -> attacker
        # controls the format string. High-precision: printf(buf) not printf("...", buf).
        "regex": re.compile(r"\bprintf\s*\(\s*[A-Za-z_]\w*\s*\)"),
        "cwe": "CWE-134",  # uncontrolled format string
        "severity": "high",
        "why": "printf() called with a non-literal format string lets input control conversions (%n, %s).",
        "what": "Calls printf() with a variable (non-literal) format string.",
        "risk": "When the format string is attacker-controlled, conversion specifiers like %x/%s/%n "
                "let an attacker read the stack or write to memory.",
        "vuln": "Uncontrolled format string",
        "recs": [
            'Use a literal format string: printf("%s", buf) instead of printf(buf).',
            "Never pass attacker-influenced data as the format argument of a *printf function."
        ],
    },
    # --- additional patterns (auto-curated from vuln-family expert pass) -----
    {
        "id": "fprintf_format_string",
        "regex": re.compile(r'''\b(?:fprintf|dprintf|syslog)\s*\(\s*[^,;\n]+,\s*[A-Za-z_]\w*\s*\)'''),
        "cwe": "CWE-134",
        "severity": "high",
        "why": "If the format argument is attacker-influenced, conversion specifiers like %x/%s/%n let an attacker read the stack or write to memory, leading to crashes or code execution.",
        "what": "Calls fprintf()/dprintf()/syslog() with a bare variable (non-literal) as the format argument and no further varargs.",
        "risk": "If the format argument is attacker-influenced, conversion specifiers like %x/%s/%n let an attacker read the stack or write to memory, leading to crashes or code execution.",
        "vuln": "Uncontrolled format string",
        "recs": [
            "Use a literal format string: fprintf(stderr, \"%s\", buf) instead of fprintf(stderr, buf).",
            "Never pass attacker-influenced data as the format argument of any *printf/syslog function.",
        ],
    },
    {
        "id": "snprintf_format_string",
        "regex": re.compile(r'''\bsnprintf\s*\(\s*[^,;\n]+,\s*[^,;\n]+,\s*[A-Za-z_]\w*\s*\)'''),
        "cwe": "CWE-134",
        "severity": "high",
        "why": "An attacker-controlled format string passed to snprintf() enables %n-style memory writes and %s/%x stack disclosure even though the output length is bounded.",
        "what": "Calls snprintf() with a bare variable (non-literal) as the format argument and no further varargs.",
        "risk": "An attacker-controlled format string passed to snprintf() enables %n-style memory writes and %s/%x stack disclosure even though the output length is bounded.",
        "vuln": "Uncontrolled format string",
        "recs": [
            "Use a literal format string: snprintf(buf, n, \"%s\", data) instead of snprintf(buf, n, data).",
            "Treat any externally derived format argument as untrusted and route the data through %s.",
        ],
    },
    {
        "id": "hardcoded_credentials",
        "regex": re.compile(r'''(?i)\b(?:passwd|password|secret|api_?key|access_?key|secret_?key|private_?key|client_?secret|passphrase|auth_?token|access_?token|credential)\b\s*(?:\[[^\]]*\])?\s*=\s*"[^"]{3,}"'''),
        "needs_strings": True,  # the credential literal is the evidence
        "cwe": "CWE-798",
        "severity": "high",
        "why": "Credentials baked into source ship in the binary and version control; anyone with read access can extract them, and rotating a compromised secret requires a code change and redeploy.",
        "what": "Assigns a non-empty string literal to a variable whose name denotes a password, secret, key, or token.",
        "risk": "Credentials baked into source ship in the binary and version control; anyone with read access can extract them, and rotating a compromised secret requires a code change and redeploy.",
        "vuln": "Use of hard-coded credentials",
        "recs": [
            "Load secrets at runtime from environment variables, a secrets manager, or a protected config file.",
            "Remove the literal from history and rotate the exposed credential immediately.",
        ],
    },
    {
        "id": "vsprintf_unbounded",
        "regex": re.compile(r'''\bvsprintf\s*\('''),
        "cwe": "CWE-120",
        "severity": "high",
        "why": "vsprintf() writes the fully formatted output with no knowledge of the destination capacity; an over-long result (often from attacker-controlled fields forwarded through a varargs wrapper) overflows the buffer.",
        "what": "Formats a va_list of arguments into a destination buffer with vsprintf(), which takes no size limit.",
        "risk": "vsprintf() writes the fully formatted output with no knowledge of the destination capacity; an over-long result (often from attacker-controlled fields forwarded through a varargs wrapper) overflows the buffer.",
        "vuln": "Buffer overflow (unbounded variadic formatting)",
        "recs": [
            "Replace vsprintf() with vsnprintf() and pass the destination buffer size.",
            "Validate or cap the length of any attacker-influenced arguments before formatting.",
        ],
    },
    {
        "id": "wcscpy_wcscat",
        "regex": re.compile(r'''\bwcs(?:cpy|cat)\s*\('''),
        "cwe": "CWE-120",
        "severity": "high",
        "why": "Like strcpy/strcat but for wide strings: the call writes until the source terminator with no bound, so a source longer than the destination overflows it (by wchar_t units, often 2-4x the byte count).",
        "what": "Copies or concatenates a wide (wchar_t) string with wcscpy()/wcscat(), neither of which checks the destination size.",
        "risk": "Like strcpy/strcat but for wide strings: the call writes until the source terminator with no bound, so a source longer than the destination overflows it (by wchar_t units, often 2-4x the byte count).",
        "vuln": "Wide-character buffer overflow (unbounded copy/concat)",
        "recs": [
            "Use wcsncpy()/wcsncat() (and ensure NUL-termination) or a bounded helper like wcslcpy/wcslcat where available.",
            "Track wide-buffer capacities explicitly and validate source length against the destination.",
        ],
    },
    {
        "id": "stpcpy_unbounded",
        "regex": re.compile(r'''\bstpcpy\s*\('''),
        "cwe": "CWE-120",
        "severity": "high",
        "why": "stpcpy() performs the same unbounded copy as strcpy() with no destination size check; a longer source overflows the destination buffer.",
        "what": "Copies a NUL-terminated string into a destination with stpcpy(), returning a pointer to the copied terminator.",
        "risk": "stpcpy() performs the same unbounded copy as strcpy() with no destination size check; a longer source overflows the destination buffer.",
        "vuln": "Buffer overflow (unbounded string copy via stpcpy)",
        "recs": [
            "Replace stpcpy() with stpncpy() (then guarantee NUL-termination) or a size-aware helper such as strlcpy().",
            "Validate that the source length is less than the destination capacity before copying.",
        ],
    },
    {
        "id": "bcopy_legacy",
        "regex": re.compile(r'''\bbcopy\s*\('''),
        "cwe": "CWE-120",
        "severity": "medium",
        "why": "bcopy() is deprecated and, like memcpy/memmove, does no bounds checking; if the length is attacker-influenced or not validated against both buffer sizes it over-reads the source or overflows the destination.",
        "what": "Copies a number of bytes between buffers using the obsolete BSD bcopy() function.",
        "risk": "bcopy() is deprecated and, like memcpy/memmove, does no bounds checking; if the length is attacker-influenced or not validated against both buffer sizes it over-reads the source or overflows the destination.",
        "vuln": "Buffer overflow / use of obsolete copy API (bcopy)",
        "recs": [
            "Replace bcopy() with memmove() (overlap-safe) or memcpy(), and validate the length against both buffer sizes.",
            "Track buffer capacities explicitly instead of raw pointers plus length.",
        ],
    },
    {
        "id": "getwd_unbounded",
        "regex": re.compile(r'''\bgetwd\s*\('''),
        "cwe": "CWE-120",
        "severity": "high",
        "why": "getwd() has no way to bound the path length, so a working-directory path longer than the destination buffer overflows it; the function is deprecated for exactly this reason.",
        "what": "Reads the current working directory path into a caller-supplied buffer using getwd(), which takes no buffer-size argument.",
        "risk": "getwd() has no way to bound the path length, so a working-directory path longer than the destination buffer overflows it; the function is deprecated for exactly this reason.",
        "vuln": "Buffer overflow via unbounded getwd()",
        "recs": [
            "Replace getwd(buf) with getcwd(buf, size), passing the destination buffer size.",
            "Use a buffer of at least PATH_MAX and check getcwd()'s return value for truncation/error.",
        ],
    },
    {
        "id": "realpath_fixed_buf",
        "regex": re.compile(r'''\brealpath\s*\([^,)]+,\s*(?!NULL\b|nullptr\b|0\s*\))[A-Za-z_]'''),
        "cwe": "CWE-120",
        "severity": "medium",
        "why": "The output buffer must hold up to PATH_MAX bytes, but PATH_MAX is not reliably the true filesystem limit; a resolved path longer than the fixed buffer overflows it. Passing NULL (auto-allocate) avoids this.",
        "what": "Resolves a pathname with realpath() and writes the canonical path into a caller-provided (non-NULL) output buffer.",
        "risk": "The output buffer must hold up to PATH_MAX bytes, but PATH_MAX is not reliably the true filesystem limit; a resolved path longer than the fixed buffer overflows it. Passing NULL (auto-allocate) avoids this.",
        "vuln": "Buffer overflow via realpath() into a fixed buffer",
        "recs": [
            "Call realpath(path, NULL) so the library allocates a correctly sized buffer, then free() it.",
            "If a fixed buffer is required, ensure it is at least PATH_MAX and confirm the platform bounds resolution to that size.",
        ],
    },
    {
        "id": "scanf_family_unbounded",
        "regex": re.compile(r'''\b(?:sscanf|fscanf|vsscanf|vfscanf|vscanf|scanf)\s*\([^;]*(?:%s|%\[)'''),
        "needs_strings": True,  # %s / %[ live inside the format literal
        "cwe": "CWE-120",
        "severity": "high",
        "why": "A %s or %[...] conversion without a width reads an arbitrarily long token into a fixed-size buffer, overflowing it; this covers the cases (e.g. sscanf(s, \"%d %s\", ...)) the bare scanf(\"%s\") rule misses.",
        "what": "Parses input with a scanf-family function (sscanf/fscanf/scanf/...) using a %s or %[ conversion that has no field-width limit.",
        "risk": "A %s or %[...] conversion without a width reads an arbitrarily long token into a fixed-size buffer, overflowing it; this covers the cases (e.g. sscanf(s, \"%d %s\", ...)) the bare scanf(\"%s\") rule misses.",
        "vuln": "Buffer overflow via *scanf %s/%[ without a field width",
        "recs": [
            "Add an explicit field width matching the buffer, e.g. sscanf(s, \"%63s\", buf) for a 64-byte buffer.",
            "Prefer fgets()/getline() with an explicit size, or parse into a std::string, instead of fixed buffers.",
        ],
    },
    {
        "id": "malloc_strlen_no_nul",
        "regex": re.compile(r'''\bmalloc\s*\(\s*strlen\s*\(\s*\w+\s*\)\s*\)'''),
        "cwe": "CWE-131",
        "severity": "medium",
        "why": "A subsequent strcpy/strcat/sprintf into the buffer writes the trailing NUL one byte past the end of the allocation, corrupting the heap (a classic off-by-one).",
        "what": "Allocates exactly strlen(s) bytes for a string buffer, with no +1 for the terminating NUL.",
        "risk": "A subsequent strcpy/strcat/sprintf into the buffer writes the trailing NUL one byte past the end of the allocation, corrupting the heap (a classic off-by-one).",
        "vuln": "Off-by-one: malloc(strlen(s)) omits the NUL terminator",
        "recs": [
            "Allocate strlen(s) + 1 bytes to leave room for the NUL terminator.",
            "Prefer strdup() (which sizes correctly) when duplicating a string.",
        ],
    },
    {
        "id": "alloc_mul_overflow",
        "regex": re.compile(r'''\bmalloc\s*\([^;)\n]*(?:\b\w+\s*\*\s*\w|\bsizeof\s*\([^)\n]*\)\s*\*|\*\s*sizeof\b)'''),
        "cwe": "CWE-190",
        "severity": "high",
        "why": "The multiplication can wrap around SIZE_MAX, so malloc() returns a buffer far smaller than the caller expects; subsequent writes that assume the full element count then overflow the heap.",
        "what": "Allocates heap memory with malloc() where the size is computed from a multiplication (e.g. count * element_size or sizeof(T) * n).",
        "risk": "The multiplication can wrap around SIZE_MAX, so malloc() returns a buffer far smaller than the caller expects; subsequent writes that assume the full element count then overflow the heap.",
        "vuln": "Integer overflow in malloc size argument",
        "recs": [
            "Use calloc(count, size) or reallocarray(), which reject overflowing element*size products.",
            "Validate that count <= SIZE_MAX/size before multiplying, or use a builtin checked-multiply (__builtin_mul_overflow).",
        ],
    },
    {
        "id": "realloc_mul_overflow",
        "regex": re.compile(r'''\brealloc\s*\([^;)\n]*(?:\b\w+\s*\*\s*\w|\bsizeof\s*\([^)\n]*\)\s*\*|\*\s*sizeof\b)'''),
        "cwe": "CWE-190",
        "severity": "high",
        "why": "If the product overflows, realloc() shrinks the allocation instead of growing it; code that keeps writing the expected number of elements overflows the now-undersized buffer.",
        "what": "Resizes a heap buffer with realloc() where the new size is a multiplication such as n * element_size.",
        "risk": "If the product overflows, realloc() shrinks the allocation instead of growing it; code that keeps writing the expected number of elements overflows the now-undersized buffer.",
        "vuln": "Integer overflow in realloc size argument",
        "recs": [
            "Use reallocarray(ptr, count, size) which fails safely on overflow.",
            "Check count against SIZE_MAX/size (or use __builtin_mul_overflow) before calling realloc().",
        ],
    },
    {
        "id": "new_array_mul_overflow",
        "regex": re.compile(r'''\bnew\s+[\w:<>]+\s*\[[^\]\n]*\b\w+\s*\*\s*\w'''),
        "cwe": "CWE-190",
        "severity": "high",
        "why": "The product is computed in the dimension's integer type and can overflow before operator new[] sees it, producing a tiny array; loops that iterate width*height then write out of bounds (heap overflow).",
        "what": "Allocates a C++ array with new T[expr] where the element count is a multiplication (e.g. new Pixel[width * height]).",
        "risk": "The product is computed in the dimension's integer type and can overflow before operator new[] sees it, producing a tiny array; loops that iterate width*height then write out of bounds (heap overflow).",
        "vuln": "Integer overflow in C++ array-new length",
        "recs": [
            "Compute the element count in a wide checked type and verify it before allocating.",
            "Prefer std::vector or std::make_unique<T[]>(n) and validate the dimensions against a sane maximum.",
        ],
    },
    {
        "id": "memcpy_mul_len_overflow",
        # The first two argument classes EXCLUDE commas so the comma distribution
        # is unambiguous — three comma-permissive greedy stars caused cubic (ReDoS)
        # backtracking on lines like memcpy(a,a,...,a,x). See tests for the guard.
        "regex": re.compile(r'''\bmemcpy\s*\([^;(),\n]*,[^;(),\n]*,[^;)\n]*(?:\b\w+\s*\*\s*\w|\bsizeof\s*\([^)\n]*\)\s*\*|\*\s*sizeof\b)'''),
        "cwe": "CWE-190",
        "severity": "high",
        "why": "If the length product overflows it wraps to a small value (mismatching the source/destination sizing logic) or, if the destination was sized from a different computation, copies far more than the destination holds — a heap/stack buffer overflow.",
        "what": "Calls memcpy() whose length argument is a multiplication such as count * element_size.",
        "risk": "If the length product overflows it wraps to a small value (mismatching the source/destination sizing logic) or, if the destination was sized from a different computation, copies far more than the destination holds — a heap/stack buffer overflow.",
        "vuln": "Integer overflow in memcpy length",
        "recs": [
            "Compute the byte length in a checked/wide type and verify it fits both buffers before the copy.",
            "Derive the destination size from the same overflow-checked count*size used for the memcpy length.",
        ],
    },
    {
        "id": "alloc_size_underflow",
        "regex": re.compile(r'''\bmalloc\s*\(\s*[\w.\)\]]+\s*-(?!>)\s*[\w(]'''),
        "cwe": "CWE-191",
        "severity": "medium",
        "why": "Allocation sizes are unsigned (size_t); if the subtrahend exceeds the minuend the result wraps to a near-SIZE_MAX value, causing either a denial-of-service-sized allocation or, when the wrapped value is later truncated, an undersized buffer.",
        "what": "Allocates with malloc() where the size is a subtraction such as end - start or len - n.",
        "risk": "Allocation sizes are unsigned (size_t); if the subtrahend exceeds the minuend the result wraps to a near-SIZE_MAX value, causing either a denial-of-service-sized allocation or, when the wrapped value is later truncated, an undersized buffer.",
        "vuln": "Unsigned integer underflow in allocation size",
        "recs": [
            "Verify that the minuend is >= the subtrahend before subtracting (e.g. check end >= start).",
            "Perform the subtraction in a signed/wide type and reject negative or oversized results before allocating.",
        ],
    },
    {
        "id": "alloc_size_from_atoi",
        "regex": re.compile(r'''\b(?:malloc|calloc|realloc|alloca)\s*\([^;)\n]*\b(?:atoi|atol|atoll)\s*\('''),
        "cwe": "CWE-789",
        "severity": "high",
        "why": "atoi() performs no validation: a huge value triggers an uncontrolled large allocation (DoS), while a negative value converts to a near-SIZE_MAX size_t — either exhausting memory or, if the result is truncated elsewhere, producing an undersized buffer.",
        "what": "Computes an allocation size from atoi()/atol()/atoll() applied to attacker-controlled input.",
        "risk": "atoi() performs no validation: a huge value triggers an uncontrolled large allocation (DoS), while a negative value converts to a near-SIZE_MAX size_t — either exhausting memory or, if the result is truncated elsewhere, producing an undersized buffer.",
        "vuln": "Allocation size taken directly from unvalidated user input",
        "recs": [
            "Parse with strtoul()/strtoll() and check errno and an explicit upper/lower bound before allocating.",
            "Clamp the requested size to a sane maximum and reject negative or zero values.",
        ],
    },
    {
        "id": "exec_path_search",
        "regex": re.compile(r'''\b(?:exec(?:lp|vpe|vp)|posix_spawnp)\s*\('''),
        "cwe": "CWE-426",
        "severity": "high",
        "why": "These variants resolve the program name against the PATH environment variable; if PATH is attacker-influenced (or the program runs setuid), a malicious binary earlier in PATH is executed instead of the intended one, yielding arbitrary code execution.",
        "what": "Launches a program by name (not absolute path) using a PATH-searching exec variant (execlp/execvp/execvpe) or posix_spawnp.",
        "risk": "These variants resolve the program name against the PATH environment variable; if PATH is attacker-influenced (or the program runs setuid), a malicious binary earlier in PATH is executed instead of the intended one, yielding arbitrary code execution.",
        "vuln": "Untrusted search path via PATH-searching exec",
        "recs": [
            "Use the non-searching variants execve()/posix_spawn() with an absolute, validated program path.",
            "If PATH search is unavoidable, sanitize PATH to a fixed trusted value before the call and never inherit it from an untrusted environment.",
        ],
    },
    {
        "id": "dlopen_untrusted_lib",
        "regex": re.compile(r'''\bdlopen\s*\('''),
        "cwe": "CWE-426",
        "severity": "low",
        "why": "If the library name is relative or derived from untrusted input, the loader resolves it through search paths (RPATH/LD_LIBRARY_PATH/cwd); an attacker who controls the path or those directories gets arbitrary code executed inside the process.",
        "what": "Loads a shared object at runtime with dlopen().",
        "risk": "If the library name is relative or derived from untrusted input, the loader resolves it through search paths (RPATH/LD_LIBRARY_PATH/cwd); an attacker who controls the path or those directories gets arbitrary code executed inside the process.",
        "vuln": "Dynamic library loading from an untrusted path",
        "recs": [
            "Pass an absolute, validated path to dlopen() and avoid relative or input-derived library names.",
            "Harden library resolution (sanitized LD_LIBRARY_PATH, trusted RPATH, verify ownership/permissions of the loaded file).",
        ],
    },
    {
        "id": "weak_crypto_primitive",
        "regex": re.compile(r'''\b(?:MD5|MD4|MD2|SHA1)\w*\s*\(|\b(?:md5|md4|md2|sha1)\w*\s*\(|\bDES_\w+\s*\(|\bRC4(?:_\w+)?\s*\(|\bRC2(?:_\w+)?\s*\(|\bEVP_(?:md5|md4|md2|sha1|rc4|rc2|des(?:_\w+)?)\s*\('''),
        "cwe": "CWE-327",
        "severity": "medium",
        "why": "MD5/SHA-1 are collision-vulnerable and unfit for signatures or password hashing; DES/RC2/RC4 are broken ciphers — using any of them undermines integrity, authentication, or confidentiality.",
        "what": "Invokes a known-weak hash or cipher primitive (MD2/MD4/MD5, SHA-1, single DES, RC2, RC4, or their OpenSSL EVP_* variants).",
        "risk": "MD5/SHA-1 are collision-vulnerable and unfit for signatures or password hashing; DES/RC2/RC4 are broken ciphers — using any of them undermines integrity, authentication, or confidentiality.",
        "vuln": "Use of a broken or risky cryptographic algorithm",
        "recs": [
            "Use SHA-256/SHA-3/BLAKE2 for hashing and AES-GCM/ChaCha20-Poly1305 for encryption.",
            "For passwords use a memory-hard KDF (argon2id, scrypt, bcrypt) rather than a raw hash.",
        ],
    },
    {
        "id": "unix_crypt_weak_hash",
        "regex": re.compile(r'''(?<![\w.>:])crypt\s*\('''),
        "cwe": "CWE-327",
        "severity": "low",
        "why": "Depending on the salt prefix, crypt() defaults to legacy DES or unsalted/low-cost MD5 hashing that is fast to brute-force, so leaked hashes are readily cracked.",
        "what": "Hashes a password using the Unix crypt() function.",
        "risk": "Depending on the salt prefix, crypt() defaults to legacy DES or unsalted/low-cost MD5 hashing that is fast to brute-force, so leaked hashes are readily cracked.",
        "vuln": "Weak password hashing via crypt()",
        "recs": [
            "Use a deliberately slow, salted KDF: argon2id, scrypt, or bcrypt with an appropriate cost factor.",
            "If crypt() must be used, force a strong scheme/cost via a modern salt (e.g. the $argon2/$y$ formats) and a high work parameter.",
        ],
    },
    {
        "id": "ecb_mode_cipher",
        "regex": re.compile(r'''\bEVP_\w*_ecb\s*\('''),
        "cwe": "CWE-327",
        "severity": "medium",
        "why": "ECB encrypts identical plaintext blocks to identical ciphertext blocks, leaking data patterns and providing no semantic security or integrity.",
        "what": "Selects an OpenSSL block cipher in ECB mode (EVP_*_ecb()).",
        "risk": "ECB encrypts identical plaintext blocks to identical ciphertext blocks, leaking data patterns and providing no semantic security or integrity.",
        "vuln": "Use of insecure ECB cipher mode",
        "recs": [
            "Use an authenticated mode such as AES-GCM (EVP_aes_256_gcm) or AES-CBC with a separate MAC and a random IV.",
            "Never use ECB for any data with structure or repeated content.",
        ],
    },
    {
        "id": "insecure_prng",
        "regex": re.compile(r'''\b(?:rand_r|rand|srand|random|srandom|drand48|lrand48|mrand48|srand48)\s*\('''),
        "cwe": "CWE-330",
        "severity": "low",
        "why": "These generators are statistically predictable and (with srand) seed-recoverable, so values used for keys, tokens, session ids, nonces, salts, or passwords can be guessed or reproduced by an attacker.",
        "what": "Generates or seeds pseudo-random values with the C library rand()/srand()/random()/drand48 family.",
        "risk": "These generators are statistically predictable and (with srand) seed-recoverable, so values used for keys, tokens, session ids, nonces, salts, or passwords can be guessed or reproduced by an attacker.",
        "vuln": "Use of cryptographically weak pseudo-random generator",
        "recs": [
            "For security use a CSPRNG: getrandom()/getentropy(), arc4random_buf(), /dev/urandom, or RAND_bytes() (OpenSSL).",
            "In C++ never use rand() for secrets; std::random_device alone is also not guaranteed CSPRNG-grade for keys.",
        ],
    },
    {
        "id": "insecure_secret_comparison",
        # The operand prefix excludes " and ( so a secret keyword only counts as an
        # actual operand — not inside a public string literal (strncmp(h,"password:",9))
        # or a sizeof() size argument (memcmp(a,b,sizeof(digest))).
        "regex": re.compile(r'''\b(?:strcmp|strncmp|strcasecmp|memcmp|bcmp)\s*\(\s*[^;\n)"(]*(?:password|passwd|secret|hmac|signature|digest|passphrase|api_?key|auth_?token|csrf_?token|session_?token)'''),
        "cwe": "CWE-208",
        "severity": "medium",
        "why": "strcmp/memcmp short-circuit on the first differing byte, so their timing leaks how many leading bytes matched; an attacker can recover the secret/MAC byte-by-byte via a timing side channel.",
        "what": "Compares a secret-like value (password, HMAC, signature, digest, token, API key) using strcmp/memcmp.",
        "risk": "strcmp/memcmp short-circuit on the first differing byte, so their timing leaks how many leading bytes matched; an attacker can recover the secret/MAC byte-by-byte via a timing side channel.",
        "vuln": "Non-constant-time comparison of a secret value",
        "recs": [
            "Use a constant-time comparison: CRYPTO_memcmp() (OpenSSL), sodium_memcmp(), or timingsafe_bcmp().",
            "Compare full-length fixed-size buffers and never branch on the result before the whole comparison completes.",
        ],
    },
    {
        "id": "world_writable_perms",
        # The octal mode must be a comma-delimited argument ending in a world-write
        # bit (2/3/6/7) followed by ',' or ')'. This ignores incidental octal tokens
        # like an array index (chmod(files[02], 0644)) and matches 5-digit setuid/
        # sticky modes (04777) that the old 4-char cap missed.
        "regex": re.compile(r'''\b(?:l?chmod|fchmod(?:at)?|mkdir(?:at)?|mkfifo|mknod)\s*\([^;]*,\s*0[0-7]*[2367]\s*[,)]'''),
        "cwe": "CWE-732",
        "severity": "high",
        "why": "Granting write permission to 'other' lets any local user modify or replace the file/directory contents, enabling tampering, privilege escalation, or planting of malicious executables.",
        "what": "Sets filesystem permissions (chmod/mkdir/mkfifo/mknod) with an octal mode whose other-bits include write (e.g. 0777, 0666, 0002).",
        "risk": "Granting write permission to 'other' lets any local user modify or replace the file/directory contents, enabling tampering, privilege escalation, or planting of malicious executables.",
        "vuln": "World-writable permissions assigned to a file or directory",
        "recs": [
            "Drop the world-writable bits: use restrictive modes such as 0644 for files or 0755/0700 for directories.",
            "Set permissions via symbolic owner/group constants and rely on a tight umask instead of broad octal literals.",
        ],
    },
    {
        "id": "insecure_temp_file",
        "regex": re.compile(r'''\b(?:mktemp|tmpnam|tempnam)\s*\('''),
        "cwe": "CWE-377",
        "severity": "medium",
        "why": "These functions only return a name and do not atomically create the file, so an attacker can predict or pre-create the path (e.g. via a symlink) between name generation and open, causing the program to clobber or read an attacker-controlled file.",
        "what": "Generates a temporary file name with mktemp()/tmpnam()/tempnam().",
        "risk": "These functions only return a name and do not atomically create the file, so an attacker can predict or pre-create the path (e.g. via a symlink) between name generation and open, causing the program to clobber or read an attacker-controlled file.",
        "vuln": "Insecure temporary file creation (predictable name / TOCTOU)",
        "recs": [
            "Use mkstemp()/mkostemp() (or tmpfile()) which atomically create the file with O_EXCL and safe permissions.",
            "Create temp files in a per-user private directory and never reuse a separately generated name.",
        ],
    },
    {
        "id": "access_toctou",
        "regex": re.compile(r'''(?<![\w.>:])access\s*\('''),
        "cwe": "CWE-367",
        "severity": "medium",
        "why": "access() tests permissions using the real (not effective) UID and the check is not atomic with the subsequent open; an attacker can swap the path (symlink race) between check and use, bypassing the guard or causing a setuid program to operate on the wrong file.",
        "what": "Checks file accessibility with access() before later opening or using the path.",
        "risk": "access() tests permissions using the real (not effective) UID and the check is not atomic with the subsequent open; an attacker can swap the path (symlink race) between check and use, bypassing the guard or causing a setuid program to operate on the wrong file.",
        "vuln": "TOCTOU / privilege check via access()",
        "recs": [
            "Drop the access() pre-check and instead open the file directly, handling the resulting error (use O_NOFOLLOW / openat with a trusted directory fd).",
            "Make security decisions on the opened file descriptor (fstat) rather than on a path that can change.",
        ],
    },
    {
        "id": "setuid_privilege_change",
        "regex": re.compile(r'''\bset(?:e|re|res)?[ug]id\s*\('''),
        "cwe": "CWE-273",
        "severity": "low",
        "why": "If the return value is ignored or the drop order (gid before uid) is wrong, the process may continue running with elevated privileges after believing it dropped them, so any later flaw executes as root.",
        "what": "Changes the process real/effective/saved user or group ID (setuid/seteuid/setgid/setresuid, etc.).",
        "risk": "If the return value is ignored or the drop order (gid before uid) is wrong, the process may continue running with elevated privileges after believing it dropped them, so any later flaw executes as root.",
        "vuln": "Privilege change without verifying success",
        "recs": [
            "Always check the return value of every set*id() call and abort on failure.",
            "Drop group privileges and supplementary groups before the user ID, then re-verify with getresuid()/getresgid() that the drop actually took effect.",
        ],
    },
    # --- C++ / lifetime additions (context-confirmed patterns) ---------------
    {
        "id": "new_delete_mismatch",
        "regex": re.compile(r"\bdelete\s*(?:(\[\s*\])\s*|\s+)([A-Za-z_]\w*)"),
        "confirm": _confirm_new_delete_mismatch,
        "cwe": "CWE-762",
        "severity": "high",
        "why": "Memory from new[] released with delete (or new released with delete[]) is undefined behavior and corrupts the allocator.",
        "what": "Releases a pointer with a delete form that does not match how it was allocated: new T[...] freed with plain delete, or a scalar new freed with delete[].",
        "risk": "The mismatched form skips (or fabricates) element destructors and hands the allocator a block layout it did not create, corrupting heap metadata — a classic primitive for heap exploitation.",
        "vuln": "Mismatched new[]/delete deallocation",
        "recs": [
            "Pair new[] with delete[] and scalar new with delete, always.",
            "Prefer std::vector / std::unique_ptr<T[]> so the correct deallocation form is automatic.",
        ],
    },
    {
        "id": "free_then_use",
        "regex": re.compile(r"\bfree\s*\(\s*([A-Za-z_]\w*)\s*\)"),
        "confirm": _confirm_free_then_use,
        "cwe": "CWE-416",
        "severity": "high",
        "why": "The pointer is dereferenced after free() without being reassigned — a use-after-free.",
        "what": "Calls free() on a pointer and later dereferences the same pointer (p->, *p, or p[...]) without reassigning it first.",
        "risk": "The freed block can be reallocated and attacker-controlled by the time it is read or written, turning the stale dereference into information disclosure or arbitrary memory corruption.",
        "vuln": "Use after free",
        "recs": [
            "Set the pointer to NULL immediately after free() and check before reuse.",
            "Restructure ownership so each allocation has one clear release point (or use RAII/smart pointers in C++).",
        ],
    },
    {
        "id": "double_free",
        "regex": re.compile(r"\bfree\s*\(\s*([A-Za-z_]\w*)\s*\)"),
        "confirm": _confirm_double_free,
        "cwe": "CWE-415",
        "severity": "high",
        "why": "The same pointer is passed to free() twice without an intervening reassignment.",
        "what": "Calls free() twice on the same pointer variable with no reassignment in between.",
        "risk": "Freeing a block twice corrupts the allocator's free lists; attackers routinely turn double frees into overlapping allocations and from there into arbitrary write primitives.",
        "vuln": "Double free",
        "recs": [
            "Set the pointer to NULL after free(); free(NULL) is a safe no-op.",
            "Give every allocation exactly one owner responsible for releasing it.",
        ],
    },
    {
        "id": "use_after_move",
        "regex": re.compile(r"\bstd\s*::\s*move\s*\(\s*([A-Za-z_]\w*)\s*\)"),
        "confirm": _confirm_use_after_move,
        "cwe": "CWE-672",
        "severity": "medium",
        "why": "The object is read again after std::move() handed its contents to another owner.",
        "what": "Passes an object through std::move() and then uses the same variable again before reassigning or re-initialising it.",
        "risk": "A moved-from object is in a valid but unspecified state: reads return garbage or empty values and invariants the code assumes (non-null handles, sizes) may no longer hold, causing logic errors or crashes.",
        "vuln": "Use of a moved-from object",
        "recs": [
            "Do not touch a variable after std::move() until it is reassigned or re-initialised.",
            "Scope moved-from variables tightly (move as the last use, e.g. in a return or emplace).",
        ],
    },
    {
        "id": "dangling_cstr",
        "regex": re.compile(
            r"(?:\bconst\s+)?\bchar\s*\*\s*(?:const\s+)?[A-Za-z_]\w*\s*=\s*[^;\n]*?\.\s*c_str\s*\(\s*\)"
            r"|\breturn\s+([A-Za-z_]\w*)\s*\.\s*c_str\s*\(\s*\)\s*;"),
        "confirm": _confirm_dangling_cstr,
        "cwe": "CWE-416",
        "severity": "medium",
        "why": "The raw pointer from .c_str() only lives as long as the owning std::string; storing or returning it invites a dangling read.",
        "what": "Stores the pointer returned by std::string::c_str() in a named char* variable, or returns a local string's c_str() from the function.",
        "risk": "The pointer is invalidated as soon as the string is mutated, reallocated, or destroyed (immediately, for a returned local), so later reads walk freed or reused heap memory.",
        "vuln": "Dangling pointer from std::string::c_str()",
        "recs": [
            "Keep the std::string itself alive and call .c_str() at the point of use instead of caching the pointer.",
            "Return std::string (or std::string_view with documented lifetime) rather than a local's c_str().",
        ],
    },
    {
        "id": "string_view_temporary",
        "regex": re.compile(
            r"\b(?:std\s*::\s*)?string_view\s+[A-Za-z_]\w*\s*=\s*[^;\n]*?(?:\bstring\s*\(|\)\s*\.\s*substr\s*\()"),
        "cwe": "CWE-416",
        "severity": "medium",
        "why": "Binding a std::string_view to a temporary std::string leaves the view dangling when the temporary is destroyed at the end of the statement.",
        "what": "Initialises a std::string_view from a temporary std::string (an explicit std::string(...) construction or a .substr() result of a call).",
        "risk": "The temporary's buffer is freed at the end of the full expression, so every later use of the view reads destroyed memory — silently returning garbage or crashing.",
        "vuln": "Dangling std::string_view over a destroyed temporary",
        "recs": [
            "Bind the std::string to a named variable that outlives the view, then take the view from it.",
            "Pass string_view only down the call stack; never store one built from a temporary.",
        ],
    },
    {
        "id": "iterator_invalidation",
        "regex": re.compile(r"\bfor\s*\(\s*[^;()\n]*:\s*([A-Za-z_]\w*)\s*\)"),
        "confirm": _confirm_iterator_invalidation,
        "cwe": "CWE-672",
        "severity": "medium",
        "why": "Mutating a container inside its own range-for loop invalidates the iterators the loop is built on.",
        "what": "Iterates a container with a range-based for loop while the loop body calls push_back/insert/erase/emplace on that same container.",
        "risk": "Growth can reallocate the backing store and erase shifts elements, so the loop's hidden iterators dangle — reads go through freed memory and writes corrupt the heap (undefined behavior that often survives testing).",
        "vuln": "Iterator invalidation (container mutated during iteration)",
        "recs": [
            "Collect changes in a second container and apply them after the loop.",
            "Use index-based iteration with explicit bounds refresh, or the erase(remove_if(...)) idiom.",
        ],
    },
    {
        "id": "vprintf_format",
        "regex": re.compile(
            r"\bvprintf\s*\(\s*[A-Za-z_]\w*\s*,"
            r"|\b(?:vfprintf|vsyslog)\s*\(\s*[^,;\n]+,\s*[A-Za-z_]\w*\s*,"
            r"|\bvsnprintf\s*\(\s*[^,;\n]+,\s*[^,;\n]+,\s*[A-Za-z_]\w*\s*,"),
        "cwe": "CWE-134",
        "severity": "high",
        "why": "A v*printf function called with a variable format string lets input control conversions (%n, %s).",
        "what": "Calls vprintf()/vfprintf()/vsnprintf()/vsyslog() with a bare variable (non-literal) as the format argument.",
        "risk": "These varargs wrappers are how attacker data most often reaches a format string: %x/%s/%n conversions then read the stack or write to memory, leading to crashes or code execution.",
        "vuln": "Uncontrolled format string (v*printf family)",
        "recs": [
            "Pass a literal format: vfprintf(fp, \"%s\", ap-rendered-text) style, never the raw variable.",
            "Audit varargs wrapper functions: mark them with __attribute__((format(printf, ...))) so the compiler checks callers.",
        ],
    },
    {
        "id": "strncpy_no_nul",
        "regex": re.compile(r"\bstrncpy\s*\(\s*([A-Za-z_](?:->|[\w.\[\]])*)\s*,"),
        "confirm": _confirm_strncpy_no_nul,
        # The trigger runs on fully masked text (a log string mentioning
        # strncpy(...) must not fire); only the CONFIRM needs literals kept, to
        # see the '\0' terminator store that makes the copy safe.
        "confirm_needs_strings": True,
        "cwe": "CWE-170",
        "severity": "medium",
        "why": "strncpy() does not NUL-terminate when the source fills the buffer, and no explicit terminator store follows.",
        "what": "Copies a string with strncpy() without any later dst[...] = '\\0' (or = 0) statement terminating the destination.",
        "risk": "When the source is as long as the size limit the destination is left unterminated, so the next strlen/strcpy/printf reads past the buffer — leaking adjacent memory or overflowing a later copy.",
        "vuln": "Missing NUL termination after strncpy",
        "recs": [
            "Follow every strncpy(dst, src, n) with dst[n-1] = '\\0'; (or use strlcpy/snprintf which terminate).",
            "Treat all fixed-size string buffers as length+1 and reserve the last byte for the terminator.",
        ],
    },
    {
        "id": "malloc_no_null_check",
        "regex": re.compile(
            r"\b([A-Za-z_]\w*)\s*=\s*(?:\([^()\n]*\)\s*)?\b(?:malloc|calloc|realloc)\s*\("),
        "confirm": _confirm_malloc_no_null_check,
        "cwe": "CWE-476",
        "severity": "medium",
        "why": "The allocation result is dereferenced immediately with no NULL check anywhere in the function.",
        "what": "Assigns the result of malloc/calloc/realloc to a pointer and dereferences it within the next few lines, with no NULL test for that pointer in the snippet.",
        "risk": "Under memory pressure (or an attacker-driven huge size) the allocator returns NULL and the unchecked dereference crashes the process — or, with an offset, writes near address zero where mapped pages make it exploitable.",
        "vuln": "NULL pointer dereference of an unchecked allocation",
        "recs": [
            "Check the allocation result (if (!p) ...) before the first dereference and fail cleanly.",
            "Centralise allocation in a checked helper (xmalloc-style) so no call site can forget the test.",
        ],
    },
    {
        "id": "strtok_nonreentrant",
        "regex": re.compile(r"\bstrtok\s*\("),
        "cwe": "CWE-676",
        "severity": "low",
        "why": "strtok() keeps hidden global state, so it is non-reentrant and unsafe with threads or nested tokenising.",
        "what": "Tokenises a string with strtok(), which stores its position in hidden static state.",
        "risk": "Concurrent or nested strtok() calls silently interleave that shared state, producing tokens from the wrong string — parsing bugs that can bypass input validation; it also writes NULs into the source buffer.",
        "vuln": "Use of non-reentrant strtok()",
        "recs": [
            "Use strtok_r() (POSIX) or strtok_s() (C11 Annex K) with an explicit save-pointer.",
            "In C++ prefer std::string_view-based splitting that does not mutate the source.",
        ],
    },
]

GENERIC_REC = [
    "Replace unsafe C APIs (gets/strcpy/strcat/sprintf) with bounded alternatives (fgets/strlcpy/strlcat/snprintf).",
    "Add explicit length checks before copying, concatenating, or formatting data, especially if attacker‑controlled.",
    "Prefer safer input routines (fgets, getline, scanf with width limits) with maximum sizes.",
    "Run classic static analyzers (clang‑tidy, cppcheck, flawfinder, etc.) alongside this classifier for structural issues.",
]

CONF_LEVELS = ["very low", "low", "medium", "high"]


def score_to_confidence(score: float, has_pattern: bool):
    """
    Map raw classifier score (0–1) + presence of a concrete bad API
    to a qualitative confidence level.
    """
    if score is None:
        return {"level": "unknown", "score": None}

    # Base bucket purely from model score
    if score >= 0.80:
        idx = 3  # high
    elif score >= 0.60:
        idx = 2  # medium
    elif score >= 0.50:
        idx = 1  # low
    else:
        idx = 0  # very low / likely noise

    # If we actually matched a dangerous API, bump confidence up one notch
    if has_pattern and idx < len(CONF_LEVELS) - 1:
        idx += 1

    return {"level": CONF_LEVELS[idx], "score": float(score)}


_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _primary_pattern(matched_ids):
    """The single most relevant matched pattern dict, or None.

    Selection: highest severity, ties broken by PATTERNS (priority) order. CWE,
    severity, and the headline vulnerability name are all read off this one
    pattern so they describe the SAME issue and cannot disagree.
    """
    matched = set(matched_ids or ())
    best, best_rank = None, -1
    for pat in PATTERNS:
        if pat["id"] in matched:
            r = _SEVERITY_RANK.get(pat.get("severity", ""), 2)
            if r > best_rank:  # strict > keeps the FIRST pattern on a tie
                best, best_rank = pat, r
    return best


def cwe_for_patterns(matched_ids):
    """CWE id of the primary matched pattern ("" when nothing maps)."""
    p = _primary_pattern(matched_ids)
    return p.get("cwe", "") if p else ""


def severity_for_patterns(matched_ids, default="medium"):
    """Severity of the primary matched pattern (the highest-severity match).

    `default` is used only when nothing matched, so a finding whose sole match is
    a 'low' pattern stays 'low'.
    """
    p = _primary_pattern(matched_ids)
    return (p.get("severity") or default) if p else default


# Default narrative used when no concrete unsafe API pattern is present (e.g. a
# region the ML classifier flagged on learned structure rather than a known API).
_GENERIC_STRUCTURED = {
    "what_code_does": "No classic unsafe C/C++ API signature was detected in this region by the "
                      "heuristic; it is flagged on the classifier's learned structure or for manual review.",
    "what_could_go_wrong": "Subtle issues that simple patterns cannot catch may remain — verify "
                           "bounds checks, input validation, integer arithmetic, and pointer "
                           "lifetimes by hand.",
    "vulnerability": "",
}


def structured_explanation(matched_ids):
    """Build the three human-facing description fields from matched pattern ids.

    Returns a dict with:
      "what_code_does"      -> "What the code is doing:"
      "what_could_go_wrong" -> "What could go wrong:"
      "vulnerability"       -> the identified vulnerability class(es)

    Multiple matches are joined in PATTERNS (priority) order. With no matches the
    generic manual-review narrative is returned.
    """
    matched = set(matched_ids or ())
    ordered = [p for p in PATTERNS if p["id"] in matched]
    if not ordered:
        return dict(_GENERIC_STRUCTURED)

    def _join(key, sep=" "):
        out, seen = [], set()
        for p in ordered:
            v = (p.get(key) or "").strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return sep.join(out)

    return {
        "what_code_does": _join("what"),
        "what_could_go_wrong": _join("risk"),
        # Distinct vuln classes read better separated than run together.
        "vulnerability": _join("vuln", sep="; "),
    }


def primary_vuln(matched_ids):
    """The primary matched pattern's vulnerability name ("" if none).

    Used for concise one-line headlines; drawn from the same primary pattern as
    cwe_for_patterns / severity_for_patterns so the headline, CWE, and severity
    all describe one coherent issue.
    """
    p = _primary_pattern(matched_ids)
    return p.get("vuln", "") if p else ""


def _dedup(seq):
    """Order-preserving dedup."""
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def explain_snippet_detailed(snippet: str):
    """Return (explanations, recommendations, matched_ids, matches).

    `matches` is a sorted list of (pattern_id, line_no) tuples with line_no
    1-based within the snippet — the anchors heuristic_scan turns into absolute
    match_lines. Regexes run over comment-masked text (string contents also
    masked unless the pattern sets "needs_strings"), so commented-out code and
    log strings cannot match; masking is same-shape, so line numbers align with
    the original snippet. `ccr-ignore` lines suppress matches here too (same
    semantics as heuristic_scan), so re-explaining a snippet cannot resurrect
    a rule the scan already suppressed.
    """
    masked_all = _mask_all(snippet)
    masked_keep_strings = _mask_code(snippet, mask_strings=False)
    raw_lines = snippet.splitlines()

    explanations, recs = [], []
    matched_ids = set()
    matches = []

    for pat in PATTERNS:
        text = masked_keep_strings if pat.get("needs_strings") else masked_all
        # Confirms that must inspect literal contents get the strings-kept
        # variant; same-shape masking keeps the trigger match's offsets valid.
        ctext = masked_keep_strings if pat.get("confirm_needs_strings") else text
        confirm = pat.get("confirm")
        for m in pat["regex"].finditer(text):
            if confirm is not None and not confirm(ctext, m):
                continue
            line_no = text.count("\n", 0, m.start()) + 1
            raw_line = raw_lines[line_no - 1] if line_no <= len(raw_lines) else ""
            rules = _suppressed_rules(raw_line)
            if rules is not None and (not rules or pat["id"] in rules):
                continue  # ccr-ignore'd on this line
            if pat["id"] not in matched_ids:
                matched_ids.add(pat["id"])
                explanations.append(pat["why"])
                recs.extend(pat["recs"])
            matches.append((pat["id"], line_no))

    if not explanations:
        explanations.append(
            "Heuristic review only: no classic unsafe APIs detected in this snippet. "
            "Inspect bounds checks, input validation, and pointer lifetimes manually."
        )
        recs.extend(GENERIC_REC)

    return _dedup(explanations), _dedup(recs), matched_ids, sorted(matches)


def explain_snippet(snippet: str):
    """
    Return (explanations, recommendations, matched_ids)
    explanations/recs are tightly tied to patterns we actually see in the snippet.
    """
    explanations, recs, matched_ids, _matches = explain_snippet_detailed(snippet)
    return explanations, recs, matched_ids


def write_html(explanations, html_path: str):
    """
    Render a simple static HTML report for human consumption.
    `explanations` is the list we emit to JSON.
    """
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)

    rows = []
    for item in explanations:
        # `or` (not a get default) so an explicit null in externally-supplied JSON
        # coerces to a safe value instead of crashing .get(...).get(...) below.
        file_path = item.get("file") or "?"
        start = item.get("start_line", "?")
        end = item.get("end_line", "?")
        score = item.get("score")
        conf = item.get("confidence") or {}
        conf_level = conf.get("level", "unknown")
        conf_score = conf.get("score")

        snippet_lines = item.get("snippet", [])
        if isinstance(snippet_lines, str):
            snippet_lines = snippet_lines.splitlines()
        code = html.escape("\n".join(snippet_lines))

        expl_list = item.get("explanations", [])
        rec_list = item.get("recommendations", [])

        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else html.escape(str(score))
        conf_score_str = f" ({conf_score:.3f})" if isinstance(conf_score, (int, float)) else ""

        # Structured 3-part narrative (preferred). Rendered whenever any of the
        # fields is present; falls back to the legacy "why/recommendations" lists.
        sev = str(item.get("severity", "")).strip()
        cwe = str(item.get("cwe", "")).strip()
        badges = []
        if sev:
            badges.append(f'<span class="badge sev-{html.escape(sev.lower())}">{html.escape(sev.upper())}</span>')
        if cwe:
            badges.append(f'<span class="badge cwe">{html.escape(cwe)}</span>')
        badge_html = " ".join(badges)

        narrative = ""
        for label, key in (("What the code is doing", "what_code_does"),
                           ("What could go wrong", "what_could_go_wrong"),
                           ("Vulnerability", "vulnerability")):
            val = str(item.get(key, "") or "").strip()
            if val:
                narrative += f"<h3>{label}</h3>\n<p>{html.escape(val)}</p>\n"

        if not narrative:
            narrative = (f"<h3>Why this is suspicious</h3>\n<ul>"
                         + "".join(f"<li>{html.escape(e)}</li>" for e in expl_list)
                         + "</ul>")

        fix = str(item.get("fix", "") or "").strip()
        if fix:
            narrative += f"<h3>Recommended fix</h3>\n<p>{html.escape(fix)}</p>\n"
        elif rec_list:
            narrative += ("<h3>Recommendations</h3>\n<ul>"
                          + "".join(f"<li>{html.escape(r)}</li>" for r in rec_list)
                          + "</ul>")

        rows.append(f"""
        <section class="finding">
          <h2>{html.escape(str(file_path))}:{html.escape(str(start))}-{html.escape(str(end))} {badge_html}</h2>
          <p><strong>Model score:</strong> {score_str} &nbsp;
             <strong>Confidence:</strong> {html.escape(conf_level)}{conf_score_str}</p>
          <pre><code>{code}</code></pre>
          {narrative}
        </section>
        """)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>LLM Vulnerability Findings</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; }}
  h1 {{ border-bottom: 1px solid #ccc; padding-bottom: .5rem; }}
  section.finding {{ border: 1px solid #ddd; padding: 1rem; margin-bottom: 1.5rem; border-radius: 6px; }}
  section.finding h3 {{ margin: .8rem 0 .2rem; font-size: .95rem; color: #333; }}
  pre {{ background: #f5f5f5; padding: .75rem; overflow-x: auto; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
  .badge {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: .1rem .45rem;
            border-radius: 4px; vertical-align: middle; margin-left: .3rem; }}
  .badge.cwe {{ background: #eef; color: #225; }}
  .sev-critical {{ background: #b00020; color: #fff; }}
  .sev-high {{ background: #d9534f; color: #fff; }}
  .sev-medium {{ background: #f0ad4e; color: #222; }}
  .sev-low {{ background: #dfe6ee; color: #333; }}
  .sev-info {{ background: #e8e8e8; color: #555; }}
</style>
</head>
<body>
<h1>LLM Vulnerability Findings</h1>
<p>Total findings: {len(explanations)}</p>
{"".join(rows)}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[OK] HTML report -> {html_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", help="classifier_findings.json from local_vuln_scanner")
    ap.add_argument("--inp", dest="inp_flag", help="Alternative to the positional input path")
    ap.add_argument("--out", required=True, help="llm_findings_heuristic.json (enriched)")
    ap.add_argument("--html", help="Optional: write an HTML report to this path")
    args = ap.parse_args()

    inp = args.inp_flag or args.inp
    if not inp:
        ap.error("input file required (positional or --inp)")

    data = json.load(open(inp, "r", encoding="utf-8"))
    findings = data.get("findings", [])

    explanations_out = []
    for f in findings:
        raw_snippet = f.get("snippet", "")
        expl, recs, matched_ids = explain_snippet(raw_snippet)
        score = f.get("score", None)
        conf = score_to_confidence(score, has_pattern=bool(matched_ids))
        structured = structured_explanation(matched_ids)

        # store snippet as list of lines (good for JSON + HTML)
        snippet_lines = raw_snippet.splitlines()[:40]

        entry = {
            "file": f.get("file"),
            "start_line": f.get("start_line"),
            "end_line": f.get("end_line"),
            "score": score,
            "confidence": conf,
            "cwe": cwe_for_patterns(matched_ids),
            "severity": severity_for_patterns(matched_ids) if matched_ids else "info",
            "matched_patterns": sorted(matched_ids),
            "what_code_does": structured["what_code_does"],
            "what_could_go_wrong": structured["what_could_go_wrong"],
            "vulnerability": structured["vulnerability"],
            "explanations": expl,
            "snippet": snippet_lines,
            "recommendations": recs,
        }
        if "match_lines" in f:  # anchor lines from heuristic_scan pass through
            entry["match_lines"] = f["match_lines"]
        explanations_out.append(entry)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"explanations": explanations_out}, f, indent=2)
    print("[OK] JSON explanations ->", args.out)

    if args.html:
        write_html(explanations_out, args.html)


if __name__ == "__main__":
    main()

