"""Conservative Malay standalone cardinal-number normalization.

Supports standard cardinals from ``kosong`` through 999,999,999,999,999,999
(below 10**18), using the Malaysian short-scale meanings ``bilion`` = 10**9
and ``trilion`` = 10**12. CLDR spellout forms and scale composition are used;
the explicit ``satu ribu`` form is also accepted as recommended by DBP in
formal contexts. Ordinals, dates, fractions, decimals written in words,
spoken negative forms, currency, inflected/dialectal forms, and larger scales
are outside scope.
Malformed scale phrases are preserved intact, and adjacent simple cardinals
are kept separate. This opt-in, context-free normalizer can still convert
cardinal homonyms such as ``satu`` when they occur in ordinary prose.

References checked: Unicode CLDR Malay RBNF
(https://github.com/unicode-org/cldr/blob/main/common/rbnf/ms.xml), the CLDR
number-formatting specification (https://unicode.org/reports/tr35/tr35-numbers.html),
DBP's Malay spelling guide (https://eseminar.dbp.gov.my/dokumen/ejaan.pdf),
and DBP's PRPM entries for zero, large scales, and number grouping
(https://prpm.dbp.gov.my/Cari1?d=7120&keyword=sifar,
https://prpm.dbp.gov.my/Cari1?d=7120&keyword=ini,
https://prpm.dbp.gov.my/Cari1?d=7120&keyword=bilion, and
https://prpm.dbp.gov.my/Cari1?d=175768&keyword=seratus).
"""

_UNITS = {
    "kosong": 0,
    "sifar": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "lapan": 8,
    "sembilan": 9,
}
_TEENS = {"sepuluh": 10, "sebelas": 11}
_GROUP_WORDS = {"belas", "puluh", "ratus"}
_SCALES = {
    "ribu": 10**3,
    "juta": 10**6,
    "bilion": 10**9,
    "trilion": 10**12,
    "kuadrilion": 10**15,
}
_IMPLICIT_SCALES = {"seribu": 10**3, "sejuta": 10**6}
_UNSUPPORTED_SCALES = {"kuintilion"}
_NON_CARDINAL_WORDS = {
    # CLDR's spoken negative form: do not normalize its cardinal operand alone.
    "negatif",
    # Fractions, fractional-number terminology, and spoken decimals.
    "per", "bahagi", "suku", "setengah", "separuh", "seperdua",
    "perdua", "sepertiga", "pertiga", "perempat", "seperempat",
    "perlima", "seperlima", "perenam", "seperenam", "pertujuh",
    "sepertujuh", "perdelapan", "seperdelapan", "persembilan",
    "sepersembilan", "persepuluh", "sepersepuluh", "perpuluhan",
    # Ordinals / collective forms, excluded from the cardinal-only mode.
    "ke", "pertama", "kedua", "ketiga", "keempat", "kelima", "keenam",
    "ketujuh", "kelapan", "kesembilan", "kesepuluh", "kesebelas",
    "kedua-dua", "satunya",
    # Currency and month names: do not partially rewrite excluded expressions.
    "ringgit", "sen", "januari", "februari", "mac", "april", "mei",
    "jun", "julai", "ogos", "september", "oktober", "november",
    "disember",
}
_NUMBER_WORDS = (
    set(_UNITS) | set(_TEENS) | _GROUP_WORDS | set(_SCALES)
    | set(_IMPLICIT_SCALES) | _UNSUPPORTED_SCALES | _NON_CARDINAL_WORDS
    | {"seratus"}
)
_MAX_VALUE = 10**18 - 1


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
            return 10 * _UNITS[first]

    if len(words) == 3 and words[1] == "puluh":
        first, _, last = words
        if (first in _UNITS and _UNITS[first] >= 2
                and last in _UNITS and _UNITS[last] > 0):
            return 10 * _UNITS[first] + _UNITS[last]
    return None


def _parse_under_1000(words):
    if not words:
        return None
    if len(words) == 1 and words[0] == "seratus":
        return 100

    if len(words) >= 2 and words[-1] == "ratus":
        coefficient_words = words[:-1]
        if coefficient_words == ["satu"]:
            return 100
        if (len(coefficient_words) == 1
                and coefficient_words[0] in _UNITS
                and _UNITS[coefficient_words[0]] >= 2):
            return _UNITS[coefficient_words[0]] * 100
        return None

    if words[0] == "seratus":
        remainder = _parse_under_100(words[1:])
        return 100 + remainder if remainder is not None and remainder > 0 else None

    if len(words) >= 2 and words[1] == "ratus":
        coefficient_words = words[:1]
        if coefficient_words == ["satu"]:
            hundreds = 100
        elif coefficient_words[0] in _UNITS and _UNITS[coefficient_words[0]] >= 2:
            hundreds = _UNITS[coefficient_words[0]] * 100
        else:
            return None
        remainder = _parse_under_100(words[2:])
        if len(words) == 2:
            return hundreds
        return (hundreds + remainder
                if remainder is not None and remainder > 0 else None)

    return _parse_under_100(words)


def _parse_cardinal(words):
    if not words or any(word in _NON_CARDINAL_WORDS | _UNSUPPORTED_SCALES
                        for word in words):
        return None

    total = 0
    previous_scale = None
    group_start = 0
    saw_scale = False

    for index, word in enumerate(words):
        if word in _IMPLICIT_SCALES:
            if group_start != index:
                return None
            scale = _IMPLICIT_SCALES[word]
            coefficient = 1
        elif word in _SCALES:
            scale = _SCALES[word]
            coefficient = _parse_under_1000(words[group_start:index])
            if coefficient is None or coefficient == 0:
                return None
        else:
            continue

        if previous_scale is not None and scale >= previous_scale:
            return None
        total += coefficient * scale
        previous_scale = scale
        group_start = index + 1
        saw_scale = True

    if not saw_scale:
        return _parse_under_1000(words)

    if group_start < len(words):
        remainder = _parse_under_1000(words[group_start:])
        if remainder is None or remainder == 0:
            return None
        total += remainder
    return total if total <= _MAX_VALUE else None


def _parse_run(run):
    if not run:
        return []
    if any(word in _NON_CARDINAL_WORDS | _UNSUPPORTED_SCALES for word in run):
        return run

    value = _parse_cardinal(run)
    if value is not None:
        return [str(value)]

    # A run containing a place-value marker is one attempted compound. Keep
    # malformed or out-of-order compounds whole rather than rewriting a prefix.
    if any(word in _GROUP_WORDS or word in _SCALES
           or word in _IMPLICIT_SCALES for word in run):
        return run

    # Without composition markers, adjacent single-word cardinals are
    # independent values (for example, a spoken list: ``dua tiga``).
    values = [_parse_under_1000([word]) for word in run]
    if all(value is not None for value in values):
        return [str(value) for value in values]
    return run


def numbers_to_digits(words):
    """Replace supported Malay cardinal runs with decimal digit tokens."""
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
