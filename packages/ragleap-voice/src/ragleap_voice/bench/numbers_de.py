"""German cardinal-number normalization for benchmark text.

Supports standard standalone cardinal forms through Billiarde (10**15),
including conventional compounds below one million and long-scale names.
Ordinals, decimals, dates, currency, other inflected forms, and regional
spellings are outside scope. Lexical forms and spellout patterns follow Unicode CLDR's
German RBNF data; large-number spelling and the German long scale are checked
against Duden. No external word list is bundled at runtime.
"""

_UNITS = {
    "null": 0, "eins": 1, "zwei": 2, "drei": 3, "vier": 4,
    "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
}
_TEENS = {
    "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13,
    "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
    "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
}
_TENS = {
    "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50,
    "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90,
}
_SMALL_SCALES = {
    "tausend": 10**3, "million": 10**6, "millionen": 10**6,
    "milliarde": 10**9, "milliarden": 10**9,
    "billion": 10**12, "billionen": 10**12,
    "billiarde": 10**15, "billiarden": 10**15,
}
_SINGULAR = {"million", "milliarde", "billion", "billiarde"}
_PLURAL = {"millionen", "milliarden", "billionen", "billiarden"}


def _spell_under_1000(number, *, final=True, hundred_and=False):
    if number == 0:
        return ""
    if number < 10:
        if number == 1 and not final:
            return "ein"
        return "eins" if number == 1 else next(w for w, n in _UNITS.items() if n == number)
    if number < 20:
        return next(w for w, n in _TEENS.items() if n == number)
    if number < 100:
        unit, ten = number % 10, number // 10 * 10
        return ((("ein" if unit == 1 else next(w for w, n in _UNITS.items() if n == unit))
                 if unit else "")
                + ("und" if unit else "")
                + next(w for w, n in _TENS.items() if n == ten))
    hundred, remainder = divmod(number, 100)
    prefix = ("ein" if hundred == 1 else next(w for w, n in _UNITS.items() if n == hundred)) + "hundert"
    if not remainder:
        return prefix
    return prefix + ("und" if hundred_and else "") + _spell_under_1000(remainder, final=final)


_UNDER_1000 = {}
for _number in range(1, 1000):
    _UNDER_1000[_spell_under_1000(_number)] = _number
    _UNDER_1000[_spell_under_1000(_number, hundred_and=True)] = _number
_UNDER_1000["hundert"] = 100
for _number in range(101, 200):
    _UNDER_1000[_spell_under_1000(_number).removeprefix("ein")] = _number
    _UNDER_1000[_spell_under_1000(_number, hundred_and=True).removeprefix("ein")] = _number
_thousand_prefixes = []
for _value in range(1, 1000):
    _prefix = _spell_under_1000(_value, final=False) + "tausend"
    _thousand_prefixes.append((_prefix, _value))
    if 100 <= _value < 200:
        _thousand_prefixes.append((_prefix.removeprefix("ein"), _value))
_THOUSAND_PREFIXES = sorted(
    _thousand_prefixes, key=lambda item: len(item[0]), reverse=True,
)


def _parse_under_1000(word):
    # Accept CLDR spellout forms with either optional "ein" before hundert/
    # tausend and optional "und" after a hundred.
    word = word.replace("ss", "ß")  # Unicode casefold maps German ß to ss.
    if word in _UNDER_1000:
        return _UNDER_1000[word]
    return 0 if word == "null" else None


def _parse_under_million(word):
    word = word.replace("ss", "ß")
    value = _parse_under_1000(word)
    if value is not None:
        return value
    # German writes compounds below one million without spaces. Try each
    # possible thousand coefficient; residual forms are independently parsed.
    for prefix, thousands in _THOUSAND_PREFIXES:
        if word.startswith(prefix):
            rest = word[len(prefix):]
            had_und = rest.startswith("und")
            if had_und:
                rest = rest[3:]
                if not rest:
                    continue
            tail = 0 if not rest else _parse_under_1000(rest)
            if tail is not None and (not rest or tail != 0):
                return thousands * 1000 + tail
    if word.startswith("tausend"):
        tail = word[len("tausend"):]
        if not tail:
            return 1000
        if tail.startswith("und"):
            tail = tail[3:]
        tail_value = _parse_under_1000(tail)
        if tail_value is None or tail_value == 0:
            return None
        return 1000 + tail_value
    return None


def _parse_scale_coefficient(words):
    """Parse a scale coefficient, including feminine ``-eine`` forms."""
    text = "".join(words).replace("ss", "ß")
    value = _parse_under_million(text)
    if value is not None:
        return value, text
    if text.endswith("eine"):
        masculine = text[:-4] + "eins"
        value = _parse_under_million(masculine)
        if value is not None:
            return value, text
    return None, text


def numbers_to_digits(words):
    """Replace recognized runs of German cardinal words with digits."""
    out, run = [], []
    for word in words:
        if (word in _SMALL_SCALES or _parse_under_million(word) is not None
                or _parse_scale_coefficient([word])[0] is not None or word == "eine"):
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
    if len(run) > 1 and all(word in _UNITS for word in run):
        return ["".join(str(_UNITS[word]) for word in run)]
    # Adjacent independent simple number words are common in lists and IDs.
    if len(run) > 1 and all(_parse_under_1000(word) is not None for word in run):
        return [str(_parse_under_1000(word)) for word in run]

    total = 0
    group = []
    previous_scale = None
    had_scale = False
    seen_large_scale = False
    thousand_coefficient = None
    for index, word in enumerate(run):
        if word not in _SMALL_SCALES:
            group.append(word)
            continue
        scale = _SMALL_SCALES[word]
        thousand_factor = (
            previous_scale == 1000 and scale > 1000 and not group
            and not seen_large_scale
        )
        if previous_scale is not None and scale >= previous_scale and not thousand_factor:
            return run
        if scale == 1000:
            coefficient = _parse_under_1000("".join(group)) if group else 1
            if group and coefficient is None:
                return run
            if group and "".join(group) == "eins":
                return run
            group = []
            thousand_coefficient = coefficient
        else:
            # Scale coefficients below one million are conventionally written
            # as one word; do not join separate words into a new number.
            if group and len(group) != 1:
                return run
            if not group and previous_scale == 1000:
                # Duden defines one Milliarde as tausend Millionen. Treat a
                # separate ``tausend`` before a larger scale as its factor.
                coefficient, coefficient_text = thousand_coefficient * 1000, ""
                total -= thousand_coefficient * 1000
            else:
                coefficient, coefficient_text = _parse_scale_coefficient(group) if group else (None, "")
            if word in _SINGULAR and (coefficient is None or not (
                    coefficient == 1 and coefficient_text == "eine"
                    or coefficient > 1 and coefficient_text.endswith("eine"))):
                return run
            if word in _PLURAL and (coefficient is None or coefficient < 2):
                return run
            group = []
        if coefficient is None or coefficient == 0:
            return run
        total += coefficient * scale
        previous_scale = scale
        had_scale = True
        if scale > 1000:
            seen_large_scale = True
    if group:
        if len(group) != 1:
            return run
        tail = _parse_under_million("".join(group))
        if tail is None or had_scale and tail == 0:
            return run
        total += tail
    elif not had_scale:
        value = _parse_under_million("".join(run))
        return [str(value)] if value is not None else run
    return [str(total)]
