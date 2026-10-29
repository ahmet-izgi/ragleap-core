"""Brazilian Portuguese cardinal-number normalization for benchmark text.

Supports standalone cardinal numbers from zero through
999,999,999,999,999,999. The vocabulary follows CLDR's Portuguese cardinal
rules, including masculine and feminine forms, and Brazilian short-scale names
through ``quadrilhão``. Malformed scale sequences are preserved. Adjacent
standalone numbers stay separate, including when joined by ``e``.

Ordinals, fractions, decimals written in words, dates, currency, negative
numbers, dialectal forms, and values of 10**18 or greater are outside scope.
Because agreement and number-versus-prose use depend on context, valid number
words such as ``um`` and ``dois`` can also be normalized in ordinary text.
No external word list is bundled or used at runtime.
"""

_UNITS = {
    "zero": 0,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "três": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
}
_TEENS = {
    "dez": 10,
    "onze": 11,
    "doze": 12,
    "treze": 13,
    "catorze": 14,
    "quatorze": 14,
    "quinze": 15,
    "dezesseis": 16,
    "dezessete": 17,
    "dezoito": 18,
    "dezenove": 19,
}
_TENS = {
    "vinte": 20,
    "trinta": 30,
    "quarenta": 40,
    "cinquenta": 50,
    "sessenta": 60,
    "setenta": 70,
    "oitenta": 80,
    "noventa": 90,
}
_HUNDREDS = {
    "duzentos": 200,
    "duzentas": 200,
    "trezentos": 300,
    "trezentas": 300,
    "quatrocentos": 400,
    "quatrocentas": 400,
    "quinhentos": 500,
    "quinhentas": 500,
    "seiscentos": 600,
    "seiscentas": 600,
    "setecentos": 700,
    "setecentas": 700,
    "oitocentos": 800,
    "oitocentas": 800,
    "novecentos": 900,
    "novecentas": 900,
}
_FEMININE_CARDINALS = {
    "uma", "duas", "duzentas", "trezentas", "quatrocentas", "quinhentas",
    "seiscentas", "setecentas", "oitocentas", "novecentas",
}
_LARGE_SCALES = {
    "milhão": 10**6,
    "milhões": 10**6,
    "bilhão": 10**9,
    "bilhões": 10**9,
    "trilhão": 10**12,
    "trilhões": 10**12,
    "quadrilhão": 10**15,
    "quadrilhões": 10**15,
}
_SINGULAR_SCALES = {"milhão", "bilhão", "trilhão", "quadrilhão"}
_UNSUPPORTED_NUMBER_MARKERS = {
    "meio", "meia", "metade", "terço", "terços", "quarto", "quartos",
    "quinto", "quintos", "sexto", "sextos", "sexta", "sextas", "sétimo",
    "sétimos", "sétima", "sétimas", "oitavo", "oitavos", "oitava",
    "oitavas", "nono", "nonos", "nona", "nonas", "décimo", "décimos",
    "décima", "décimas", "centésimo", "centésimos", "centésima",
    "centésimas", "milésimo", "milésimos", "milésima", "milésimas", "avo",
    "avos", "primeiro", "primeira", "segundo", "segunda", "terceiro",
    "terceira", "quarta", "quinta",
}
_NUMBER_WORDS = (
    set(_UNITS)
    | set(_TEENS)
    | set(_TENS)
    | set(_HUNDREDS)
    | set(_LARGE_SCALES)
    | {"cem", "cento", "mil"}
    | _UNSUPPORTED_NUMBER_MARKERS
)
_SMALL_WORDS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_HUNDREDS) | {
    "cem", "cento"
}
_MAX_VALUE = 10**18 - 1


def _parse_under_100(words):
    if len(words) == 1:
        word = words[0]
        if word in _UNITS:
            return _UNITS[word]
        if word in _TEENS:
            return _TEENS[word]
        if word in _TENS:
            return _TENS[word]
        if word == "cem":
            return 100
        return None

    if len(words) == 3 and words[1] == "e":
        tens = _TENS.get(words[0])
        unit = _UNITS.get(words[2])
        if tens is not None and unit is not None and 1 <= unit <= 9:
            return tens + unit
    return None


def _parse_under_1000(words):
    if not words:
        return None
    if len(words) == 1:
        word = words[0]
        if word == "cento":
            return None
        if word in _HUNDREDS:
            return _HUNDREDS[word]
        return _parse_under_100(words)

    under_100 = _parse_under_100(words)
    if under_100 is not None:
        return under_100

    head = words[0]
    if head == "cento":
        hundreds = 100
    else:
        hundreds = _HUNDREDS.get(head)
    if hundreds is None or len(words) < 3 or words[1] != "e":
        return None

    tail = _parse_under_100(words[2:])
    if tail is None or tail == 0 or tail >= 100:
        return None
    return hundreds + tail


def _parse_scale_coefficient(words, scale_word):
    if not words:
        return 1 if scale_word == "mil" else None
    value = _parse_under_1000(words)
    if value is None:
        return None
    if scale_word == "mil":
        # In standard Brazilian Portuguese, one thousand is simply "mil".
        if value == 1:
            return None
        return value
    if any(word in _FEMININE_CARDINALS for word in words):
        # Million and larger scale nouns are masculine; CLDR's feminine rules
        # switch back to agreement with the counted noun only after the scale.
        return None
    if value < 1 or value > 999:
        return None
    if scale_word in _SINGULAR_SCALES:
        return value if value == 1 and words == ["um"] else None
    return value if value >= 2 else None


def _parse_run(run):
    if not run:
        return None

    if not any(word in _LARGE_SCALES or word == "mil" for word in run):
        return _parse_under_1000(run)

    total = 0
    previous_scale = _MAX_VALUE + 1
    group = []
    for word in run:
        scale = 1000 if word == "mil" else _LARGE_SCALES.get(word)
        if scale is None:
            group.append(word)
            continue
        if scale >= previous_scale:
            return None
        if group and group[0] == "e":
            if total == 0:
                return None
            group = group[1:]
        coefficient = _parse_scale_coefficient(group, word)
        if coefficient is None:
            return None
        total += coefficient * scale
        previous_scale = scale
        group = []

    if group and group[0] == "e":
        group = group[1:]
    if group:
        remainder = _parse_under_1000(group)
        if remainder is None or remainder == 0:
            return None
        total += remainder
    elif run[-1] == "e":
        return None

    return total if total <= _MAX_VALUE else None


def _parse_adjacent(run):
    """Parse clearly adjacent standalone values after a compound parse fails."""
    if any(word in _LARGE_SCALES or word == "mil" for word in run):
        return None

    parts = []
    start = 0
    for index, word in enumerate(run):
        if word == "e":
            parts.append(run[start:index])
            start = index + 1
    parts.append(run[start:])
    if any(not part or _parse_under_1000(part) is None for part in parts):
        # Without a conjunction, a sequence of complete standalone values is
        # also a common way to read identifiers or lists digit by digit.
        if "e" in run or any(word not in _SMALL_WORDS for word in run):
            return None
        if any(word in _HUNDREDS for word in run):
            return None
        values = [_parse_under_1000([word]) for word in run]
        return values if all(value is not None for value in values) else None
    values = [_parse_under_1000(part) for part in parts]
    if any(value >= 100 for value in values):
        return None
    separated = []
    for index, value in enumerate(values):
        if index:
            separated.append("e")
        separated.append(value)
    return separated


def _convert_run(run):
    value = _parse_run(run)
    if value is not None:
        return [str(value)]
    adjacent = _parse_adjacent(run)
    if adjacent is not None:
        return [str(value) for value in adjacent]
    return run


def numbers_to_digits(words):
    """Replace valid Brazilian Portuguese cardinal-number runs with digits."""
    out, run = [], []
    for index, word in enumerate(words):
        next_word = words[index + 1] if index + 1 < len(words) else ""
        if word in _NUMBER_WORDS:
            run.append(word)
        elif word == "e" and run and (
            next_word in _NUMBER_WORDS
            or run[-1] in _TENS
            or run[-1] == "cento"
            or run[-1] in _HUNDREDS
            or run[-1] in _LARGE_SCALES
            or run[-1] == "mil"
        ):
            run.append(word)
        else:
            out.extend(_convert_run(run))
            run = []
            out.append(word)
    out.extend(_convert_run(run))
    return out
