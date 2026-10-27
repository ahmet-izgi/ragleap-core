"""Spanish cardinal-number normalization for benchmark text.

The supported forms are standalone cardinal number words from zero through
cuatrillón, using conventional spaced spelling, standard long-scale names,
and masculine apocopations before those scale nouns. Other gendered forms,
ordinals, dates, currency forms, optional one-word spellings of some numbers
from 31 to 99, and regional and dialect variants are outside this module's
scope.

The lexical forms and combinations follow the RAE's cardinal-numeral reference
and CLDR's Spanish rule-based number data. The RAE's dictionary supplies
millardo and the long-scale values for billón, trillón, and cuatrillón; the US
short-scale variants are not supported. No external word list is bundled or
used at runtime.
"""

_UNITS = {
    "cero": 0,
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
}
_TEENS = {
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciséis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
}
_TENS = {
    "veinte": 20,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}
_COMPOUND_TWENTIES = {
    "veintiuno": 21,
    "veintidós": 22,
    "veintitrés": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiséis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}
_MASCULINE_APOCOPE_TWENTY_ONE = "veintiún"
_HUNDREDS = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "trescientos": 300,
    "cuatrocientos": 400,
    "quinientos": 500,
    "seiscientos": 600,
    "setecientos": 700,
    "ochocientos": 800,
    "novecientos": 900,
}
_LARGE_SCALES = {
    "millón": 10 ** 6,
    "millones": 10 ** 6,
    "millardo": 10 ** 9,
    "millardos": 10 ** 9,
    "billón": 10 ** 12,
    "billones": 10 ** 12,
    "trillón": 10 ** 18,
    "trillones": 10 ** 18,
    "cuatrillón": 10 ** 24,
    "cuatrillones": 10 ** 24,
}
_SINGULAR_LARGE_SCALES = {
    "millón",
    "millardo",
    "billón",
    "trillón",
    "cuatrillón",
}
_PLURAL_LARGE_SCALES = {
    "millones": 10 ** 6,
    "millardos": 10 ** 9,
    "billones": 10 ** 12,
    "trillones": 10 ** 18,
    "cuatrillones": 10 ** 24,
}
_SCALES = {"mil": 10 ** 3, **_LARGE_SCALES}
_NUMBER_WORDS = (
    set(_UNITS)
    | set(_TEENS)
    | set(_TENS)
    | set(_COMPOUND_TWENTIES)
    | set(_HUNDREDS)
    | set(_SCALES)
)
_UNDER_100 = {**_UNITS, **_TEENS, **_TENS, **_COMPOUND_TWENTIES}


def numbers_to_digits(words):
    """Replace recognized runs of Spanish cardinal words with digits."""
    out, run = [], []
    for index, word in enumerate(words):
        next_word = words[index + 1] if index + 1 < len(words) else ""
        if word in _NUMBER_WORDS:
            run.append(word)
        elif word == _MASCULINE_APOCOPE_TWENTY_ONE and next_word in (
            _SINGULAR_LARGE_SCALES | set(_PLURAL_LARGE_SCALES) | {"mil"}
        ):
            run.append(word)
        elif word == "un" and next_word in (
            _SINGULAR_LARGE_SCALES | set(_PLURAL_LARGE_SCALES) | {"mil"}
        ):
            # Keep apocopated "un" with the following scale candidate so
            # invalid forms such as "un mil" remain unchanged as a whole.
            run.append("un")
        elif (
            word == "y"
            and run
            and (
                (
                    run[-1] in _TENS
                    and (
                        next_word in _UNITS
                        or (
                            next_word == "un"
                            and index + 2 < len(words)
                            and words[index + 2] in (
                                _SINGULAR_LARGE_SCALES | set(_PLURAL_LARGE_SCALES) | {"mil"}
                            )
                        )
                    )
                )
                # Keep malformed hundred + conjunction sequences in one run so
                # the parser leaves the complete phrase untouched.
                or (run[-1] in _HUNDREDS and next_word in _NUMBER_WORDS)
                # The conjunction only joins tens and units in standard
                # cardinal spelling, not scale words and another number.
                or (run[-1] in _SCALES and next_word in _NUMBER_WORDS)
            )
        ):
            # Spanish joins tens and units from treinta to noventa with "y".
            run.append(word)
        else:
            out.extend(_parse_run(run))
            run = []
            out.append(word)
    out.extend(_parse_run(run))
    return out


def _parse_run(run):
    if not run:
        return []

    # Consecutive unit words can be read digit by digit, as in the other modes.
    if len(run) > 1 and all(word in _UNITS for word in run):
        return ["".join(str(_UNITS[word]) for word in run)]

    out = []
    total = 0
    group = []
    last_scale = None
    used_scale = False
    index = 0

    while index < len(run):
        scale, consumed = _scale_at(run, index)
        if scale is None:
            group.append(run[index])
            index += 1
            continue

        if group == ["ciento"]:
            return run
        if run[index] == "mil" and group == ["un"]:
            # "Un mil" is regionally attested in parts of the Americas, but
            # this locale-neutral module does not select that regional form.
            return run
        if (
            run[index] in (_SINGULAR_LARGE_SCALES | set(_PLURAL_LARGE_SCALES) | {"mil"})
            and group
            and group[-1] in ("un", _MASCULINE_APOCOPE_TWENTY_ONE)
        ):
            # Masculine scale nouns trigger apocope: uno -> un and
            # veintiuno -> veintiún. Parse these forms only in this context.
            expanded_group = list(group)
            expanded_group[-1] = (
                "uno" if expanded_group[-1] == "un" else "veintiuno"
            )
            coefficient = _parse_under_1000(expanded_group)
        elif group:
            coefficient = _parse_under_1000(group)
        elif run[index] == "mil" or consumed == 2:
            coefficient = 1
        else:
            # Millón, millardo, billón, etc. are nouns and need a determiner.
            return run
        if coefficient is None:
            # Do not reinterpret an unsupported or malformed phrase.
            return run
        if coefficient == 0:
            return run
        if (
            run[index] in (set(_PLURAL_LARGE_SCALES) | {"mil"})
            and group
            and group[-1] in ("uno", "veintiuno")
        ):
            # Before masculine scale nouns, the final uno must be apocopated
            # (e.g. "veintiún mil" and "veintiún millones").
            return run
        if consumed == 1 and (
            run[index] in _SINGULAR_LARGE_SCALES
            or run[index] in _PLURAL_LARGE_SCALES
        ):
            # RAE treats these scale words as nouns: singular forms take one,
            # while plural forms take a coefficient greater than one.
            if run[index] in _SINGULAR_LARGE_SCALES and coefficient != 1:
                return run
            if run[index] in _SINGULAR_LARGE_SCALES and group != ["un"]:
                return run
            if run[index] in _PLURAL_LARGE_SCALES and coefficient <= 1:
                return run
        if last_scale is not None and scale >= last_scale:
            if total:
                out.append(str(total))
            total = 0

        total += coefficient * scale
        group = []
        last_scale = scale
        used_scale = True
        index += consumed

    tail = _parse_under_1000(group) if group else 0
    if tail is None:
        return run
    if group == ["cero"] and used_scale:
        return run
    total += tail

    if used_scale or total or run:
        out.append(str(total))
    return out


def _scale_at(words, index):
    word = words[index]
    if word == "mil" and index + 1 < len(words):
        next_word = words[index + 1]
        if next_word in _PLURAL_LARGE_SCALES:
            return 10 ** 3 * _PLURAL_LARGE_SCALES[next_word], 2
    if word in _SCALES:
        return _SCALES[word], 1
    return None, 1


def _parse_under_1000(words):
    if not words:
        return None

    value = 0
    if words[0] == "cien" and len(words) > 1:
        return None
    if words[0] in _HUNDREDS:
        value = _HUNDREDS[words[0]]
        words = words[1:]
        if words == ["cero"]:
            return None
    if not words:
        return value

    if len(words) == 1 and words[0] in _UNDER_100:
        return value + _UNDER_100[words[0]]

    if (
        len(words) == 3
        and words[0] in _TENS
        and _TENS[words[0]] >= 30
        and words[1] == "y"
        and words[2] in _UNITS
        and words[2] != "cero"
    ):
        return value + _TENS[words[0]] + _UNITS[words[2]]

    return None
