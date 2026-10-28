"""Conservative Italian cardinal-number normalization for benchmark text.

Supports standard written cardinal forms from zero through 999,999,999,999,999,999
(below ``trilione``): closed compounds through 999,999, and spaced scale nouns
from ``milione`` through ``biliardo``. Large-scale tails require the conjunction
``e``; malformed scale sequences are preserved as written. Adjacent standalone
number words are normalized separately. Ordinals, decimals, dates, currency,
inflected forms, regional/archaic spellings, and forms above the CLDR ``biliardo``
range are outside scope. Italian cardinal words can also be ordinary prose; this
opt-in normalizer does not infer their grammatical role.

Lexical forms and composition follow Unicode CLDR's Italian RBNF data, checked
against Treccani, Accademia della Crusca, and the University of Messina's DICO
Italian-language advice. No external word list is bundled at runtime.
"""

_UNITS = {
    "zero": 0, "uno": 1, "due": 2, "tre": 3, "quattro": 4,
    "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9,
}
_TEENS = {
    "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13,
    "quattordici": 14, "quindici": 15, "sedici": 16,
    "diciassette": 17, "diciotto": 18, "diciannove": 19,
}
_TENS = {
    "venti": 20, "trenta": 30, "quaranta": 40, "cinquanta": 50,
    "sessanta": 60, "settanta": 70, "ottanta": 80, "novanta": 90,
}
_HUNDREDS = {
    1: "cento", 2: "duecento", 3: "trecento", 4: "quattrocento",
    5: "cinquecento", 6: "seicento", 7: "settecento", 8: "ottocento",
    9: "novecento",
}
_LARGE_SCALES = {
    "milione": 10**6, "milioni": 10**6,
    "miliardo": 10**9, "miliardi": 10**9,
    "bilione": 10**12, "bilioni": 10**12,
    "biliardo": 10**15, "biliardi": 10**15,
}
_SINGULAR_SCALES = {"milione", "miliardo", "bilione", "biliardo"}
_PLURAL_SCALES = {"milioni", "miliardi", "bilioni", "biliardi"}
_MAX_VALUE = 10**18 - 1


def _spell_under_100(number):
    if number < 20:
        return next(word for word, value in {**_UNITS, **_TEENS}.items()
                    if value == number)
    ten, unit = divmod(number, 10)
    tens_word = next(word for word, value in _TENS.items() if value == ten * 10)
    if unit == 1:
        return tens_word[:-1] + "uno"
    if unit == 8:
        return tens_word[:-1] + "otto"
    if unit == 3:
        return tens_word + "tré"
    return tens_word + (_UNITS_WORDS[unit] if unit else "")


_UNITS_WORDS = {value: word for word, value in _UNITS.items()}


def _spell_under_1000(number):
    if number < 100:
        return _spell_under_100(number)
    hundred, remainder = divmod(number, 100)
    prefix = _HUNDREDS[hundred]
    if not remainder:
        return prefix
    rest = _spell_under_100(remainder)
    if 80 <= remainder < 90:
        # The initial o of ottanta is elided after a hundred: centottanta.
        return prefix[:-1] + rest
    return prefix + rest


_UNDER_1000 = {
    _spell_under_1000(number): number for number in range(1000)
}
# Compounds ending in standalone tre take the acute accent, including 103.
for _hundred in range(1, 10):
    _UNDER_1000[_HUNDREDS[_hundred] + "tré"] = _hundred * 100 + 3
# Both forms are documented for 180; CLDR and Crusca favor centottanta.
for _hundred in range(1, 10):
    _prefix = _HUNDREDS[_hundred]
    # CLDR writes centotto, while Crusca prefers centootto.
    _UNDER_1000[_prefix[:-1] + "otto"] = _hundred * 100 + 8
# Crusca accepts centoottanta beside its more usual centottanta.
_UNDER_1000["centoottanta"] = 180
# The rare centuno is also documented beside centouno.
_UNDER_1000["centuno"] = 101
_UNDER_1000_BEFORE_SCALE = {
    word[:-3] + "tre": value
    for word, value in _UNDER_1000.items() if word.endswith("tré")
}


def _parse_under_1000(word):
    return _UNDER_1000.get(word)


def _parse_under_million(word):
    """Parse a canonical closed Italian cardinal below one million."""
    value = _parse_under_1000(word)
    if value is not None:
        return value

    if word.startswith("mille"):
        tail = word[len("mille"):]
        if not tail:
            return 1000
        remainder = _parse_under_1000(tail)
        if remainder is not None and remainder > 0:
            return 1000 + remainder
        return None

    marker = word.rfind("mila")
    if marker <= 0:
        return None
    coefficient_word, tail_word = word[:marker], word[marker + len("mila"):]
    coefficient = (_parse_under_1000(coefficient_word)
                   if coefficient_word not in _UNDER_1000_BEFORE_SCALE
                   else _UNDER_1000_BEFORE_SCALE[coefficient_word])
    if coefficient is None and coefficient_word.endswith("un"):
        # Crusca attests both ventunmila and ventunomila.
        coefficient = _parse_under_1000(coefficient_word + "o")
    remainder = (0 if not tail_word else 3 if tail_word == "tré"
                 and coefficient > 1 else _parse_under_1000(tail_word))
    if coefficient is None or not 2 <= coefficient <= 999:
        return None
    if remainder is None:
        return None
    return coefficient * 1000 + remainder


def _parse_small_phrase(words):
    """Parse one written cardinal, including a conjunction after a round group."""
    if len(words) == 1:
        return _parse_under_million(words[0])
    if len(words) == 3 and words[1] == "e":
        base = _parse_under_million(words[0])
        tail = _parse_under_million(words[2])
        if (base is not None and base >= 100
                and (base % 100 == 0 or base % 1000 == 0)
                and tail is not None and tail > 0):
            value = base + tail
            return value if value < 10**6 else None
    return None


def _parse_scale_chain(words, previous_scale=None):
    """Parse a validated sequence such as ``un milione e due miliardi``."""
    if len(words) < 2 or words[1] not in _LARGE_SCALES:
        return None

    coefficient_word, scale_word = words[:2]
    coefficient = (1 if coefficient_word == "un"
                   else _parse_under_million(coefficient_word))
    scale = _LARGE_SCALES[scale_word]
    if coefficient is None or not 1 <= coefficient <= 999:
        return None
    if scale_word in _SINGULAR_SCALES:
        if coefficient != 1 or coefficient_word != "un":
            return None
    elif coefficient < 2:
        return None
    if previous_scale is not None and scale >= previous_scale:
        return None

    value = coefficient * scale
    if len(words) == 2:
        return value
    if words[2] != "e" or len(words) == 3:
        return None

    tail = words[3:]
    if any(word in _LARGE_SCALES for word in tail):
        remainder = _parse_scale_chain(tail, previous_scale=scale)
    else:
        remainder = _parse_small_phrase(tail)
    if remainder is None or remainder == 0:
        return None
    result = value + remainder
    return result if result <= _MAX_VALUE else None


def _is_number_word(word):
    return (word == "un" or word == "mila" or word in _LARGE_SCALES
            or _parse_under_million(word) is not None)


def _parse_run(run):
    if not run:
        return []
    if any(word in _LARGE_SCALES for word in run):
        value = _parse_scale_chain(run)
        return [str(value)] if value is not None else run
    if "mila" in run:
        return run
    if "e" in run:
        value = _parse_small_phrase(run)
        return [str(value)] if value is not None else run
    values = [_parse_under_million(word) for word in run]
    if any(value is None for value in values):
        return run
    return [str(value) for value in values]


def numbers_to_digits(words):
    """Replace recognized runs of Italian cardinal words with digits."""
    out, run = [], []
    for word in words:
        if _is_number_word(word):
            run.append(word)
        elif word == "e" and run:
            last_word = run[-1]
            last_value = _parse_under_million(last_word)
            joins_number = (
                last_word == "mille" or last_word in _LARGE_SCALES
                or last_value is not None and last_value >= 100
                and (last_value % 100 == 0 or last_value % 1000 == 0)
            )
            if joins_number:
                run.append(word)
            else:
                out.extend(_parse_run(run))
                run = []
                out.append(word)
        else:
            out.extend(_parse_run(run))
            run = []
            out.append(word)
    out.extend(_parse_run(run))
    return out
