#!/usr/bin/env python3
import argparse, os, json, re, html

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


def explain_snippet(snippet: str):
    """
    Return (explanations, recommendations, matched_ids)
    explanations/recs are tightly tied to patterns we actually see in the snippet.
    """
    explanations = []
    recs = []
    matched_ids = set()

    for pat in PATTERNS:
        if pat["regex"].search(snippet):
            matched_ids.add(pat["id"])
            explanations.append(pat["why"])
            recs.extend(pat["recs"])

    if not explanations:
        explanations.append(
            "Heuristic review only: no classic unsafe APIs detected in this snippet. "
            "Inspect bounds checks, input validation, and pointer lifetimes manually."
        )
        recs.extend(GENERIC_REC)

    # Dedup while preserving order
    def dedup(seq):
        out, seen = [], set()
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedup(explanations), dedup(recs), matched_ids


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

        explanations_out.append({
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
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"explanations": explanations_out}, f, indent=2)
    print("[OK] JSON explanations ->", args.out)

    if args.html:
        write_html(explanations_out, args.html)


if __name__ == "__main__":
    main()

