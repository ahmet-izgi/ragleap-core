"""Conservative Indonesian standalone cardinal-number normalization.

Supports standard cardinals from nol through 999,999,999,999,999 (999
triliun). Only standard cardinal spellings and descending place-value groups
are accepted; malformed phrases are preserved intact. Ordinals, fractions,
dates, currency, digit-by-digit readings (including ``kosong``), dialectal or
older spellings, the scale name ``biliun`` (whose value varies by reference),
and numbers above the documented range are outside scope.

The number words and composition are checked against CLDR's Indonesian RBNF,
the Indonesian Reference Grammar, and Tata Bahasa Baku Bahasa Indonesia.
Indonesian has no grammatical gender or number inflection on these scale
words. Context-free normalization can still rewrite ordinary prose containing
cardinal homonyms (notably ``satu``); the feature is opt-in.
"""

_UNITS = {
    "nol": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
}
_TEENS = {"sepuluh": 10, "sebelas": 11}
_GROUP_WORDS = {"belas", "puluh", "ratus"}
_SCALES = {"ribu": 10**3, "juta": 10**6, "miliar": 10**9,
           "triliun": 10**12}
_IMPLICIT_SCALES = {"seribu": "ribu", "sejuta": "juta"}
_UNSUPPORTED_SCALES = {"biliun", "milyar", "kuadriliun", "kuintiliun",
                       "sekstiliun", "septiliun", "oktiliun", "noniliun",
                       "desiliun"}
_NON_CARDINAL_WORDS = {
    "kosong", "koma", "titik", "setengah", "separuh", "seperdua",
    "sepertiga", "seperempat", "seperlima", "seperenam", "sepersepuluh",
    "seperduapuluh", "perdua", "pertiga", "perempat", "perlima", "perenam",
    "pertujuh", "perdelapan", "persembilan", "persepuluh", "persebelas",
    "perseratus", "perseribu", "persen", "permil", "satunya",
}
_NUMBER_WORDS = (set(_UNITS) | set(_TEENS) | _GROUP_WORDS | set(_SCALES)
                 | set(_IMPLICIT_SCALES) | _UNSUPPORTED_SCALES
                 | _NON_CARDINAL_WORDS | {"seratus"})
_MAX_VALUE = 10**15 - 1


def _parse_under_100(words):
    if len(words) == 1:
        word = words[0]
        if word in _UNITS:
            return _UNITS[word]
        return _TEENS.get(word)
    if len(words) == 2:
        first, second = words
        if second == "belas" and first in _UNITS and _UNITS[first] >= 2:
            return 10 + _UNITS[first]
        if second == "puluh" and first in _UNITS and _UNITS[first] >= 2:
            return _UNITS[first] * 10
        return None
    if (len(words) == 3 and words[1] == "puluh"
            and words[0] in _UNITS and 2 <= _UNITS[words[0]] <= 9
            and words[2] in _UNITS and 1 <= _UNITS[words[2]] <= 9):
        return _UNITS[words[0]] * 10 + _UNITS[words[2]]
    return None


def _parse_under_1000(words):
    if not words:
        return None
    if words[0] == "seratus":
        if len(words) == 1:
            return 100
        remainder = _parse_under_100(words[1:])
        return 100 + remainder if remainder is not None and remainder else None
    if len(words) >= 2 and words[1] == "ratus":
        coefficient = _UNITS.get(words[0])
        if coefficient is None or not 2 <= coefficient <= 9:
            return None
        if len(words) == 2:
            return coefficient * 100
        remainder = _parse_under_100(words[2:])
        return coefficient * 100 + remainder if remainder is not None and remainder else None
    return _parse_under_100(words)


def _parse_cardinal(words):
    """Parse a single cardinal phrase, returning None for unsupported forms."""
    if not words:
        return None

    # The fused forms are the standard one-thousand and one-million forms.
    expanded = []
    implicit = {}
    for word in words:
        if word in _IMPLICIT_SCALES:
            scale_word = _IMPLICIT_SCALES[word]
            implicit[len(expanded)] = scale_word
            expanded.append(scale_word)
        else:
            expanded.append(word)

    if not any(word in _SCALES for word in expanded):
        return _parse_under_1000(words)

    total = 0
    previous_scale = None
    start = 0
    saw_scale = False
    for index, word in enumerate(expanded):
        if word not in _SCALES:
            continue
        scale = _SCALES[word]
        if previous_scale is not None and scale >= previous_scale:
            return None
        coefficient_words = expanded[start:index]
        if index in implicit:
            if coefficient_words:
                return None
            coefficient = 1
        else:
            coefficient = _parse_under_1000(coefficient_words)
            if coefficient is None or coefficient == 0:
                return None
            # One thousand is seribu; one million may also be satu juta.
            # For larger scales, CLDR and the standard grammar use satu.
            if scale == 10**3 and coefficient == 1:
                return None
        total += coefficient * scale
        previous_scale = scale
        start = index + 1
        saw_scale = True

    tail_words = expanded[start:]
    if tail_words:
        tail = _parse_under_1000(tail_words)
        if tail is None or tail == 0:
            return None
        total += tail
    if not saw_scale or total > _MAX_VALUE:
        return None
    return total


def _parse_run(run):
    if not run:
        return []
    # Keep non-cardinal constructions together when their surrounding number
    # words would otherwise be converted independently.
    if any(word in _UNSUPPORTED_SCALES or word in _NON_CARDINAL_WORDS for word in run):
        return run
    value = _parse_cardinal(run)
    if value is not None:
        return [str(value)]

    # A scale-containing phrase is all-or-nothing. This prevents a repeated
    # or increasing scale from being partially rewritten as a plausible value.
    if any(word in _SCALES or word in _IMPLICIT_SCALES for word in run):
        return run

    # Consecutive standalone cardinals are separate values. Greedy longest
    # valid prefixes still allow compounds such as ``dua puluh satu``.
    # Choose the longest first phrase that still permits a complete split.
    # A greedy prefix alone would misread ``dua puluh tiga puluh`` as 23 plus
    # an orphaned ``puluh`` instead of two adjacent values, 20 and 30.
    split = {len(run): []}
    for start in range(len(run) - 1, -1, -1):
        # A supported group below one thousand has at most five tokens.
        for stop in range(min(len(run), start + 5), start, -1):
            candidate = _parse_cardinal(run[start:stop])
            if candidate is not None and stop in split:
                split[start] = [str(candidate), *split[stop]]
                break
    # Keep malformed marker-based phrases intact instead of producing partial
    # digits such as ``2 belas puluh``.
    return split.get(0, run)


def numbers_to_digits(words):
    """Replace valid Indonesian cardinal runs with decimal digit tokens."""
    out, run = [], []
    for word in words:
        if word in _NUMBER_WORDS:
            run.append(word)
        else:
            out.extend(_parse_run(run))
            run = []
            out.append(word)
    out.extend(_parse_run(run))
    return out
