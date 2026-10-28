"""The optional number step of the WER normalizer (issue #754)."""
import json
import pathlib
import sys
import wave

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ragleap_voice.bench import (
    cer,
    environment_info,
    load_manifest,
    normalize,
    normalizer_version,
    run_stt,
    summarize,
    wer,
    write_results,
)

# benchmarks/stt-neutral-10s: the ground truth, and how an engine that writes digits
# would transcribe the same audio
NEUTRAL_REF = ("Please schedule a callback for tomorrow at three fifteen. Order number four "
               "seven two nine, total due one hundred twenty dollars.")
NEUTRAL_HYP = ("Please schedule a callback for tomorrow at 3:15. Order number 4729, "
               "total due 120 dollars.")


def test_default_is_unchanged():
    """Published numbers were measured without the step, so it stays off by default."""
    assert normalizer_version() == "1"
    assert normalize("four seven two nine") == "four seven two nine"
    assert normalize("4,729") == "4 729"
    assert normalize("kırk iki") == "kırk iki"
    assert normalize("birinci") == "birinci"
    assert wer(NEUTRAL_REF, NEUTRAL_HYP) == pytest.approx(9 / 21)


def test_version_names_the_number_step():
    assert normalizer_version(numbers=True) == "1+numbers-en.1"


def test_stt_neutral_10s_example():
    assert normalize(NEUTRAL_REF, numbers=True) == (
        "please schedule a callback for tomorrow at 3 15 order number 4729 total due 120 dollars")
    assert wer(NEUTRAL_REF, NEUTRAL_HYP, numbers=True) == 0
    assert cer(NEUTRAL_REF, NEUTRAL_HYP, numbers=True) == 0



def test_spanish_number_words_and_metrics():
    # Cardinal forms and combinations follow RAE's numeral reference and
    # CLDR's Spanish RBNF data; no external word list is bundled at runtime.
    # https://www.rae.es/diccionario-estudiante/docs/numerales.pdf
    # https://dle.rae.es/millardo
    # https://dle.rae.es/bill%C3%B3n
    # https://dle.rae.es/trill%C3%B3n
    # https://dle.rae.es/cuatrill%C3%B3n
    # https://github.com/unicode-org/cldr/blob/main/common/rbnf/es.xml
    examples = [
        ("cero", "0"),
        ("uno", "1"),
        ("dos", "2"),
        ("tres", "3"),
        ("cuatro", "4"),
        ("cinco", "5"),
        ("seis", "6"),
        ("siete", "7"),
        ("ocho", "8"),
        ("nueve", "9"),
        ("diez", "10"),
        ("once", "11"),
        ("doce", "12"),
        ("trece", "13"),
        ("catorce", "14"),
        ("quince", "15"),
        ("dieciséis", "16"),
        ("diecisiete", "17"),
        ("dieciocho", "18"),
        ("diecinueve", "19"),
        ("veinte", "20"),
        ("veintiuno", "21"),
        ("veintidós", "22"),
        ("veintitrés", "23"),
        ("veinticuatro", "24"),
        ("veinticinco", "25"),
        ("veintiséis", "26"),
        ("veintisiete", "27"),
        ("veintiocho", "28"),
        ("veintinueve", "29"),
        ("treinta", "30"),
        ("treinta y uno", "31"),
        ("cuarenta", "40"),
        ("cincuenta", "50"),
        ("sesenta", "60"),
        ("setenta", "70"),
        ("ochenta", "80"),
        ("noventa", "90"),
        ("noventa y nueve", "99"),
        ("cien", "100"),
        ("ciento uno", "101"),
        ("ciento dieciséis", "116"),
        ("ciento veintidós", "122"),
        ("doscientos treinta y cuatro", "234"),
        ("trescientos", "300"),
        ("cuatrocientos", "400"),
        ("quinientos", "500"),
        ("seiscientos", "600"),
        ("setecientos", "700"),
        ("ochocientos", "800"),
        ("novecientos", "900"),
        ("novecientos noventa y nueve", "999"),
        ("mil", "1000"),
        ("mil uno", "1001"),
        ("veintiún mil", "21000"),
        ("treinta y un mil", "31000"),
        ("ciento un mil", "101000"),
        ("dos mil veintiséis", "2026"),
        ("un millón", "1000000"),
        ("veintiún millones", "21000000"),
        ("treinta y un millones", "31000000"),
        ("ciento un millones", "101000000"),
        ("un billón", "1000000000000"),
        ("un trillón", "1000000000000000000"),
        ("un cuatrillón", "1000000000000000000000000"),
        ("un millón doscientos treinta y cuatro mil quinientos sesenta y siete",
         "1234567"),
        ("mil millones", "1000000000"),
        ("un millardo", "1000000000"),
        ("dos billones tres mil millones cuatro millones cinco mil seis",
         "2003004005006"),
        ("dos mil tres millones", "2003000000"),
        ("dos mil tres millones cuatro mil cinco", "2003004005"),
        ("veintiún mil dos millones", "21002000000"),
        ("uno dos tres", "123"),  # Unit words spoken digit by digit.
        ("TOTAL: treinta y dos", "total 32"),
        ("VEINTIDÓS", "22"),
    ]
    for text, expected in examples:
        assert normalize(text, numbers="es") == expected

    assert normalize("un perro", numbers="es") == "un perro"
    assert normalize("una persona", numbers="es") == "una persona"
    assert normalize("millón millones", numbers="es") == "millón millones"
    assert normalize("un millones", numbers="es") == "un millones"
    # Leave regionally/contextually marked "un mil" forms untouched. The
    # RAE grammar notes this use in parts of America, especially financial or
    # legal documents; this normalizer does not choose a regional variant.
    assert normalize("un mil", numbers="es") == "un mil"
    assert normalize("un mil millones", numbers="es") == "un mil millones"
    # Repeated or increasing scales are malformed, so preserve the complete run.
    for malformed_scale in (
        "dos millones tres millones",
        "mil millón",
        "dos billones tres trillones",
    ):
        assert normalize(malformed_scale, numbers="es") == malformed_scale
    # Malformed/noncanonical sequences are preserved as a whole, not partly
    # rewritten into misleading values.
    for malformed in (
        "cien uno",
        "uno millones",
        "veintiuno millones",
        "treinta y uno millones",
        "ciento uno millones",
        "uno millón",
        "dos millón",
        "uno mil",
        "veintiuno mil",
        "treinta y uno mil",
        "ciento uno mil",
        "ciento cero",
        "ciento millones",
        "ciento mil",
        "cien y uno",
        "treinta y cero",
        "ciento treinta y cero",
        "cero mil",
        "mil cero",
        "dos mil cero",
        "mil y uno",
        "mil y cero",
        "dos millones y uno",
    ):
        assert normalize(malformed, numbers="es") == malformed
    assert normalize("primero", numbers="es") == "primero"
    assert normalize("doscientas", numbers="es") == "doscientas"
    # Spanish decimal commas must not be mistaken for thousands separators.
    assert normalize("1,234", numbers="es") == "1 234"
    assert normalize("1,234", numbers="en") == "1234"
    # Adjacent standalone Spanish numerals represent separate spoken values.
    assert normalize("treinta cuarenta", numbers="es") == "30 40"
    assert normalize("diez once", numbers="es") == "10 11"
    assert normalizer_version(numbers="es") == "1+numbers-es.1"

    reference = "Tengo treinta y dos años."
    hypothesis = "Tengo 32 años."
    assert normalize(reference, numbers="es") == normalize(hypothesis, numbers="es")
    assert wer(reference, hypothesis, numbers="es") == 0
    assert cer(reference, hypothesis, numbers="es") == 0


def test_german_cardinal_number_words_and_metrics():
    # German lexical forms and spellout rules follow CLDR's German RBNF data;
    # large-number spelling and long-scale values are checked against Duden.
    # https://github.com/unicode-org/cldr/blob/main/common/rbnf/de.xml
    # https://www.duden.de/sprachwissen/sprachratgeber/Schreibung-von-Zahlen
    examples = [
        ("null", "0"), ("eins", "1"), ("zwei", "2"), ("drei", "3"),
        ("vier", "4"), ("fünf", "5"), ("sechs", "6"), ("sieben", "7"),
        ("acht", "8"), ("neun", "9"), ("zehn", "10"), ("elf", "11"),
        ("zwölf", "12"), ("dreizehn", "13"), ("sechzehn", "16"),
        ("siebzehn", "17"), ("neunzehn", "19"), ("zwanzig", "20"),
        ("einundzwanzig", "21"), ("siebenundzwanzig", "27"),
        ("dreißig", "30"), ("vierzig", "40"), ("neunundneunzig", "99"),
        ("einhundert", "100"), ("hundert", "100"),
        ("einhunderteins", "101"), ("zweihundertfünfunddreißig", "235"),
        ("einhundertundfünf", "105"), ("hundertfünf", "105"),
        ("hundertfünfzig", "150"), ("neunhundertneunundneunzig", "999"),
        ("tausend", "1000"), ("eintausendeins", "1001"),
        ("tausendundeins", "1001"), ("eintausendundeins", "1001"),
        ("hunderttausend", "100000"), ("hunderttausendeins", "100001"),
        ("hundertzwanzigtausend", "120000"), ("hunderttausendfünf", "100005"),
        ("zweitausendzweihundertsechsundfünfzig", "2256"),
        ("neunhundertneunundneunzigtausendneunhundertneunundneunzig", "999999"),
        ("eine Million", "1000000"), ("zwei Millionen", "2000000"),
        ("eine Milliarde", "1000000000"), ("zwei Milliarden", "2000000000"),
        ("tausend Millionen", "1000000000"),
        ("eine Billion", "1000000000000"), ("eine Billiarde", "1000000000000000"),
        ("zwei Millionen dreihunderttausendvierhundertfünf", "2300405"),
        ("eins zwei drei", "123"), ("ZAHL: einundzwanzig", "zahl 21"),
    ]
    for text, expected in examples:
        assert normalize(text, numbers="de") == expected

    assert normalize("eine Katze", numbers="de") == "eine katze"
    assert normalize("ein Million", numbers="de") == "ein million"
    assert normalize("zwei Million", numbers="de") == "zwei million"
    assert normalize("eine Millionen", numbers="de") == "eine millionen"
    assert normalize("zwei Milliarden drei Millionen", numbers="de") == "2003000000"
    assert normalize("einhunderteine Million", numbers="de") == "101000000"
    assert normalize("einhunderteine Milliarden", numbers="de") == "101000000000"
    assert normalize("zwei Millionen dreizehn", numbers="de") == "2000013"
    assert normalize("zwei Millionen drei zehn", numbers="de") == "zwei millionen drei zehn"
    for malformed in ("zwei Millionen drei Millionen", "zwei Millionen drei Milliarden",
                      "zwei Milliarden drei Billionen",
                      "eins Millionen", "eins Million", "eins Milliarde",
                      "eins tausend", "tausend null", "zwei Millionen null",
                      "tausendnull", "eintausendnull", "zweitausendnull",
                      "eintausendund", "eine Million tausend Millionen",
                      "zwei Milliarden tausend Millionen"):
        assert normalize(malformed, numbers="de") == malformed.casefold()
    assert normalize("einundzwanzig vierzig", numbers="de") == "21 40"
    assert normalize("zwanzig drei vier", numbers="de") == "20 3 4"
    assert normalize("einundzwanzig vierzig neun", numbers="de") == "21 40 9"
    assert normalizer_version(numbers="de") == "1+numbers-de.1"
    assert normalize("1,5", numbers="de") == "1,5"
    assert normalize("3,14", numbers="de") == "3,14"
    assert normalize("1,234", numbers="de") == "1,234"
    # Keep the existing punctuation behavior for the other number modes.
    assert normalize("1,234", numbers="en") == "1234"
    assert normalize("1,234", numbers="tr") == "1234"
    assert normalize("1,234", numbers="es") == "1 234"
    assert normalize("einundzwanzig", numbers="de") == normalize("21", numbers="de")
    assert wer("einundzwanzig", "21", numbers="de") == 0
    assert cer("einundzwanzig", "21", numbers="de") == 0



def test_italian_cardinal_number_words_and_metrics():
    # CLDR supplies the spellout rules. Treccani, Crusca, and the University of
    # Messina verify Italian composition, spelling variants, and scale syntax.
    # https://raw.githubusercontent.com/unicode-org/cldr/main/common/rbnf/it.xml
    # https://raw.githubusercontent.com/unicode-org/cldr/main/common/main/it.xml
    # https://www.treccani.it/enciclopedia/numerali-prontuario_%28Enciclopedia-dell%27Italiano%29/
    # https://accademiadellacrusca.it/it/consulenza/quarantaquattro-gatti-in-fila-per-sei-col-resto-di-due-o-di-quando-e-come-scrivere-i-numeri-in-lettere/1077
    # https://dico.unime.it/ufaq/la-forma-dei-numerali-cardinali-e-ordinali/
    # https://dico.unime.it/ufaq/numeri-e-parole/
    # https://www.treccani.it/vocabolario/bilione/
    # https://www.treccani.it/enciclopedia/biliardo_%28Enciclopedia-della-Matematica%29/
    examples = [
        ("zero", "0"), ("uno", "1"), ("due", "2"), ("tre", "3"),
        ("quattro", "4"), ("cinque", "5"), ("sei", "6"),
        ("sette", "7"), ("otto", "8"), ("nove", "9"),
        ("dieci", "10"), ("undici", "11"), ("dodici", "12"),
        ("tredici", "13"), ("quattordici", "14"), ("quindici", "15"),
        ("sedici", "16"), ("diciassette", "17"), ("diciotto", "18"),
        ("diciannove", "19"), ("venti", "20"), ("ventuno", "21"),
        ("ventidue", "22"), ("ventitré", "23"), ("ventotto", "28"),
        ("trentuno", "31"), ("quarantatré", "43"),
        ("novantanove", "99"), ("cento", "100"),
        ("centotré", "103"), ("centouno", "101"), ("centuno", "101"), ("centootto", "108"),
        ("centotto", "108"), ("centottanta", "180"),
        ("centoottanta", "180"), ("duecentotré", "203"),
        ("duecentoventitré", "223"),
        ("novecentonovantanove", "999"), ("mille", "1000"),
        ("milleuno", "1001"), ("mille e uno", "1001"),
        ("cento e due", "102"), ("trecento e due", "302"),
        ("duemila e novantanove", "2099"),
        ("duemilatré", "2003"), ("duemilacentotré", "2103"),
        ("milleottocento", "1800"), ("duemila", "2000"),
        ("ventunomila", "21000"), ("ventunmila", "21000"),
        ("centoventitremilaquattrocentocinquantasei", "123456"),
        ("novecentonovantanovemilanovecentonovantanove", "999999"),
        ("un milione", "1000000"), ("due milioni", "2000000"),
        ("un milione e uno", "1000001"),
        ("un milione e novantunmila", "1091000"),
        ("due milioni e centomila", "2100000"),
        ("un miliardo", "1000000000"),
        ("due miliardi e tre", "2000000003"),
        ("due miliardi e tre milioni", "2003000000"),
        ("un bilione", "1000000000000"),
        ("due bilioni", "2000000000000"),
        ("un biliardo", "1000000000000000"),
        ("novecentonovantanove biliardi e novecentonovantanove bilioni e "
         "novecentonovantanove miliardi e novecentonovantanove milioni e "
         "novecentonovantanovemilanovecentonovantanove",
         "999999999999999999"),
        ("Ho ventitré libri.", "ho 23 libri"),
        ("VENTITRÉ", "23"),
    ]
    for text, expected in examples:
        assert normalize(text, numbers="it") == expected

    # Articles and bare scale nouns are not treated as numbers.
    assert normalize("un cane", numbers="it") == "un cane"
    assert normalize("milione", numbers="it") == "milione"
    assert normalize("un trilione", numbers="it") == "un trilione"
    # This opt-in lexical pass cannot distinguish quantities from ordinary or
    # figurative uses; document the visible false positives explicitly.
    assert normalize("grazie mille", numbers="it") == "grazie 1000"
    assert normalize("uno dei miei amici", numbers="it") == "1 dei miei amici"
    # Singular/plural agreement, conjunctions, and strictly descending scales.
    for malformed in (
        "uno milione", "due milione", "un milioni", "zero milioni",
        "ventiuno", "ventitre", "ventun", "duecentoottanta",
        "due mila", "zero mila", "mille e zero",
        "due milioni tre milioni", "due milioni e tre miliardi",
        "due miliardi e tre miliardi",
        "un milione due", "un milione e un", "un milione e due milioni tre",
    ):
        assert normalize(malformed, numbers="it") == malformed

    # Adjacent words remain separate values (unit-only runs are digit strings).
    assert normalize("uno due tre", numbers="it") == "123"
    assert normalize("venti trenta", numbers="it") == "20 30"
    assert normalize("venti e uno", numbers="it") == "20 e 1"
    assert normalize("cento due", numbers="it") == "100 2"
    assert normalize("cento uno", numbers="it") == "100 1"
    # The shared punctuation contract splits hyphens; it must not invent 21.
    assert normalize("venti-uno", numbers="it") == "20 1"
    assert normalize("vent'uno", numbers="it") == "21"
    assert normalize("ventitre\u0301", numbers="it") == "23"
    assert normalize("  ventitré   libri ", numbers="it") == "23 libri"
    assert normalize("２３", numbers="it") == "23"
    # Italian decimal commas are retained; complete period groups are thousands.
    for decimal in ("1,5", "3,14", "1,234"):
        assert normalize(decimal, numbers="it") == decimal
    assert normalize("1.234", numbers="it") == "1234"
    assert normalize("12.345", numbers="it") == "12345"
    assert normalize("12.345.678", numbers="it") == "12345678"
    assert normalize("1.234,5", numbers="it") == "1234,5"
    assert normalize("12.34", numbers="it") == "12 34"
    assert normalize("1,234", numbers="en") == "1234"

    assert normalizer_version(numbers="it") == "1+numbers-it.1"
    assert normalizer_version() == "1"
    assert normalize("ventitré", numbers=False) == "ventitré"
    assert normalize("ventitré", numbers=True) == "ventitré"

    reference = "Ho ventitré anni."
    hypothesis = "Ho 23 anni."
    assert normalize(reference, numbers="it") == normalize(hypothesis, numbers="it")
    assert wer(reference, hypothesis, numbers="it") == 0
    assert cer(reference, hypothesis, numbers="it") == 0


def test_indonesian_cardinal_number_words_and_metrics():
    # Forms and ordering follow CLDR's Indonesian RBNF and the Indonesian
    # Reference Grammar; the current standard's scales and grouping were
    # checked against Tata Bahasa Baku Bahasa Indonesia.
    # https://github.com/unicode-org/cldr/blob/main/common/rbnf/id.xml
    # https://theswissbay.ch/pdf/Books/Linguistics/Mega%20linguistics%20pack/Austronesian/Indonesian%20Reference%20Grammar%20%28Sneddon%29.pdf
    # https://repositori.kemendikdasmen.go.id/16351/1/Tata%20Bahasa%20Baku%20Bahasa%20Ind.pdf
    # https://github.com/unicode-org/cldr/blob/main/common/main/id.xml
    examples = [
        ("nol", "0"),
        ("satu", "1"),
        ("dua", "2"),
        ("tiga", "3"),
        ("empat", "4"),
        ("lima", "5"),
        ("enam", "6"),
        ("tujuh", "7"),
        ("delapan", "8"),
        ("sembilan", "9"),
        ("sepuluh", "10"),
        ("sebelas", "11"),
        ("dua belas", "12"),
        ("sembilan belas", "19"),
        ("dua puluh", "20"),
        ("dua puluh satu", "21"),
        ("sembilan puluh sembilan", "99"),
        ("seratus", "100"),
        ("seratus satu", "101"),
        ("lima ratus lima puluh empat", "554"),
        ("sembilan ratus sembilan puluh sembilan", "999"),
        ("seribu", "1000"),
        ("seribu satu", "1001"),
        ("dua ribu tiga ratus", "2300"),
        ("tujuh ribu delapan ratus lima puluh sembilan", "7859"),
        ("sejuta", "1000000"),
        ("satu juta", "1000000"),
        ("dua juta lima ratus ribu", "2500000"),
        ("tiga juta empat ratus lima puluh enam ribu tujuh ratus delapan puluh sembilan",
         "3456789"),
        ("satu miliar", "1000000000"),
        ("dua miliar tiga juta", "2003000000"),
        ("satu triliun", "1000000000000"),
        ("sembilan ratus sembilan puluh sembilan triliun sembilan ratus sembilan puluh sembilan miliar sembilan ratus sembilan puluh sembilan juta sembilan ratus sembilan puluh sembilan ribu sembilan ratus sembilan puluh sembilan",
         "999999999999999"),
        ("TOTAL: dua puluh satu", "total 21"),
        ("DUA PULUH SATU", "21"),
    ]
    for text, expected in examples:
        assert normalize(text, numbers="id") == expected

    # Adjacent numbers are separate cardinals, not a digit string.
    assert normalize("satu dua", numbers="id") == "1 2"
    assert normalize("dua puluh tiga puluh", numbers="id") == "20 30"
    assert normalize("sebelas dua belas", numbers="id") == "11 12"
    assert normalize("dua ratus nol", numbers="id") == "200 0"
    assert normalize("antara dua dan tiga", numbers="id") == "antara 2 dan 3"
    assert normalize("seratus dan satu", numbers="id") == "100 dan 1"

    # Repeated/increasing scales and structurally malformed groups are kept
    # intact rather than partially rewritten.
    for malformed in (
        "dua juta tiga juta",
        "dua miliar tiga triliun",
        "seribu juta",
        "satu ribu",
        "satu ratus",
        "satu belas",
        "dua belas puluh",
        "dua ratus puluh",
        "dua juta nol",
        "nol ribu",
        "sejuta sejuta",
    ):
        assert normalize(malformed, numbers="id") == malformed

    # Words for digit-by-digit readings, fractions, ordinals, and indefinite
    # quantities are outside the cardinal parser.
    for unsupported in ("kosong", "setengah", "sepertiga", "kesepuluh", "belasan",
                        "tiga kosong dua", "satu setengah", "dua sepertiga",
                        "dua pertiga", "tiga perempat", "dua perlima", "dua perenam",
                        "dua perdelapan", "dua persepuluh", "dua perseratus",
                        "seperenam belas", "satu persen", "satu permil",
                        "dua koma lima", "dua titik lima", "dua biliun",
                        "dua kuadriliun"):
        assert normalize(unsupported, numbers="id") == unsupported
    # Punctuation is normalized by the shared base pipeline, but neither
    # number in the mixed-number fraction is converted.
    assert normalize("tiga dua-pertiga", numbers="id") == "tiga dua pertiga"
    assert normalize("Ini satu-satunya cara", numbers="id") == "ini satu satunya cara"
    assert normalize("tidak ada angka", numbers="id") == "tidak ada angka"
    assert normalize("jumlahnya sekitar dua puluh", numbers="id") == "jumlahnya sekitar 20"

    # CLDR specifies period grouping and comma decimals for Indonesian.
    assert normalize("1.234", numbers="id") == "1234"
    assert normalize("1.234.567", numbers="id") == "1234567"
    assert normalize("1,5", numbers="id") == "1,5"
    assert normalize("1.234,5", numbers="id") == "1234,5"
    assert normalize("1.234,5.", numbers="id") == "1234,5"
    assert normalize("1.234,5.678", numbers="id") == "1 234,5 678"
    assert normalize("12.34", numbers="id") == "12 34"
    assert normalize("12.345.67", numbers="id") == "12 345 67"
    assert normalize("1.234.567.89", numbers="id") == "1 234 567 89"
    assert normalize("1,234", numbers="id") == "1,234"
    assert normalize("dua-puluh-satu!", numbers="id") == "21"
    assert normalize("dua\u00a0puluh  satu", numbers="id") == "21"
    assert normalize("ＤＵＡ ＰＵＬＵＨ ＳＡＴＵ", numbers="id") == "21"
    assert normalize("١.٢٣٤", numbers="id") == "1234"

    # Indonesian grouping must consume the entire dotted digit sequence.
    # Italian retains its pre-existing prefix grouping behavior; this PR must
    # not silently alter that language's versioned normalization.
    assert normalize("12.345.67", numbers="it") == "12345 67"
    assert normalize("1.234.567.89", numbers="it") == "1234567 89"
    assert normalize("1.234,5", numbers="it") == "1234,5"

    assert normalizer_version(numbers="id") == "1+numbers-id.1"
    assert normalizer_version() == "1"
    assert normalize("dua puluh satu") == "dua puluh satu"
    assert normalize("dua puluh satu", numbers="en") == "dua puluh satu"
    assert normalize("satu juta", numbers="de") == "satu juta"
    assert normalize("einundzwanzig", numbers="id") == "einundzwanzig"

    # Fraction words are recognized compositionally: the per-/seper- prefix
    # is protected when followed by a supported single-word denominator.
    for fraction in ("dua seperdelapan", "tiga sepertujuh", "empat persepuluh",
                     "lima persembilan", "dua perseratus"):
        assert normalize(fraction, numbers="id") == fraction
    assert normalize("buku persembahan dua", numbers="id") == "buku persembahan 2"
    assert normalize("pernol", numbers="id") == "pernol"

    reference = "Saya berusia dua puluh satu tahun."
    hypothesis = "Saya berusia 21 tahun."
    assert normalize(reference, numbers="id") == normalize(hypothesis, numbers="id")
    assert wer(reference, hypothesis, numbers="id") == 0
    assert cer(reference, hypothesis, numbers="id") == 0


def test_turkish_number_words_and_metrics():
    # TDK documents ordinal spelling and digit-suffix conventions; lexical test
    # examples are author-written and cross-checked against Turkish references.
    # https://tdk.gov.tr/icerik/yazim-kurallari/sayilarin-yazilisi/
    # No external corpus or number-word list is bundled or used at runtime.
    examples = [
        ("sıfır", "0"),
        ("yirmi bir", "21"),
        ("kırk iki", "42"),
        ("yüz yirmi üç", "123"),
        ("dokuz yüz doksan dokuz", "999"),
        ("iki bin yirmi altı", "2026"),
        ("İKİ BİN YİRMİ ALTI", "2026"),
        ("KIRK İKİ", "42"),
        ("ALTMIŞ", "60"),
        ("bir milyon iki yüz otuz dört bin beş yüz altmış yedi", "1234567"),
        ("bir iki üç", "123"),  # Unit words spoken digit by digit.
        ("Toplam kırk iki puan aldım.", "toplam 42 puan aldım"),
        ("İki yüz otuz dört kişi geldi.", "234 kişi geldi"),
    ]
    for text, expected in examples:
        assert normalize(text, numbers="tr") == expected
    assert normalize("Toplam ٤٢ puan", numbers="tr") == "toplam 42 puan"

    reference = "Bu sınavdan kırk iki puan aldım."
    hypothesis = "Bu sınavdan 42 puan aldım."
    assert normalize(reference, numbers="tr") == normalize(hypothesis, numbers="tr")
    assert wer(reference, hypothesis, numbers="tr") == 0
    assert cer(reference, hypothesis, numbers="tr") == 0


def test_turkish_ordinal_words_and_metrics():
    # Ordinal lexical forms are hand-authored; TDK supports the writing
    # conventions; Zemberek (Apache-2.0) lists sıfırıncı and katrilyonuncu,
    # while Omniglot lists katrilyoncu as an attested variant:
    # https://github.com/ahmetaa/zemberek-nlp/blob/master/morphology/src/main/resources/tr/turkish-ordinal-numbers.txt
    # https://omniglot.com/language/numbers/turkish.htm
    examples = [
        ("sıfırıncı", "0"),
        ("birinci", "1"),
        ("İKİNCİ", "2"),
        ("üçüncü", "3"),
        ("dördüncü", "4"),
        ("beşinci", "5"),
        ("altıncı", "6"),
        ("yedinci", "7"),
        ("sekizinci", "8"),
        ("dokuzuncu", "9"),
        ("onuncu", "10"),
        ("yirminci", "20"),
        ("kırkıncı", "40"),
        ("doksanıncı", "90"),
        ("yüzüncü", "100"),
        ("bininci", "1000"),
        ("milyonuncu", "1000000"),
        ("milyarıncı", "1000000000"),
        ("trilyonuncu", "1000000000000"),
        ("katrilyoncu", "1000000000000000"),
        ("katrilyonuncu", "1000000000000000"),
        ("yirmi birinci", "21"),
        ("yüz yirmi üçüncü", "123"),
        ("iki bin yirmi altıncı", "2026"),
        ("iki milyonuncu", "2000000"),
        ("Bu, yarışın üçüncü turu.", "bu yarışın 3 turu"),
        ("Birinci, ikinci ve üçüncü oldular.", "1 2 ve 3 oldular"),
    ]
    for text, expected in examples:
        assert normalize(text, numbers="tr") == expected

    reference = "Bugün yarışın yirmi birinci günü."
    hypothesis = "Bugün yarışın 21'inci günü."
    assert normalize(reference, numbers="tr") == normalize(hypothesis, numbers="tr")
    assert wer(reference, hypothesis, numbers="tr") == 0
    assert cer(reference, hypothesis, numbers="tr") == 0


@pytest.mark.parametrize("text, expected", [
    ("0'ıncı", "0"),
    ("1'inci", "1"),
    ("2'nci", "2"),
    ("3'üncü", "3"),
    ("4'üncü", "4"),
    ("15’inci", "15"),
    ("56’ncı", "56"),
    ("56'ncı sırada", "56 sırada"),
    ("1'inci/2'nci", "1 2"),
])
def test_turkish_digit_ordinal_suffixes(text, expected):
    assert normalize(text, numbers="tr") == expected


def test_turkish_version_and_english_alias():
    assert normalizer_version(numbers="tr") == "1+numbers-tr.2"
    assert normalizer_version(numbers="en") == normalizer_version(numbers=True) == "1+numbers-en.1"
    assert normalize("twenty one", numbers="en") == normalize("twenty one", numbers=True) == "21"
    assert normalize("birinci", numbers=True) == "birinci"


@pytest.mark.parametrize("numbers", ["fr", "TR", ""])
def test_unsupported_number_language_is_rejected(numbers):
    with pytest.raises(ValueError, match="unsupported number-normalization language"):
        normalize("hello", numbers=numbers)
    with pytest.raises(ValueError, match="unsupported number-normalization language"):
        normalizer_version(numbers=numbers)


def test_non_boolean_non_string_number_option_is_rejected():
    with pytest.raises(TypeError, match="numbers must be"):
        normalize("hello", numbers=1)


@pytest.mark.parametrize("text, expected", [
    ("four seven two nine", "4729"),                      # digit by digit
    ("one two", "12"),
    ("seven", "7"),
    ("fifteen", "15"),
    ("twenty one", "21"),
    ("twenty-one", "21"),
    ("one hundred twenty", "120"),
    ("one hundred and twenty", "120"),
    ("twelve hundred", "1200"),
    ("one thousand two hundred", "1200"),
    ("two million three hundred thousand", "2300000"),
    ("hundred", "100"),
    ("three fifteen", "3 15"),                            # a word that cannot continue starts a new number
    ("twenty twenty", "20 20"),
    ("one thousand one thousand", "1000 1000"),
    ("one hundred one hundred", "100 100"),
    ("one thousand one", "1001"),
    ("one hundred one", "101"),
    ("nineteen eighty", "19 80"),                         # years in pairs: not covered
    ("rock and roll", "rock and roll"),                   # "and" only joins number words
    ("one hundred and some", "100 and some"),
    ("the one I like", "the 1 i like"),
])
def test_english_number_words(text, expected):
    assert normalize(text, numbers=True) == expected


def test_run_stt_applies_turkish_to_both_texts_and_records_version(tmp_path):
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    m = tmp_path / "m.csv"
    m.write_text("id,path,reference,language,license,source\n"
                 "a,a.wav,Bu sınavdan kırk iki puan aldım.,tr,MIT,author-written MIT test\n", encoding="utf-8")
    items, problems = load_manifest(m)
    assert problems == []

    rows = run_stt(lambda path, language: "Bu sınavdan 42 puan aldım.", items, numbers="tr")
    assert summarize(rows)["tr"]["wer"] == 0
    assert summarize(rows)["tr"]["cer"] == 0
    _, json_path = write_results(tmp_path / "out", "fake", rows, environment_info())
    saved = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    assert saved["meta"]["normalizer_version"] == "1+numbers-tr.2"


def test_run_stt_applies_italian_and_records_version(tmp_path):
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    m = tmp_path / "m.csv"
    m.write_text("id,path,reference,language,license,source\n"
                 "a,a.wav,Ho ventitré anni.,it,MIT,author-written MIT test\n", encoding="utf-8")
    items, problems = load_manifest(m)
    assert problems == []

    rows = run_stt(lambda path, language: "Ho 23 anni.", items, numbers="it")
    assert summarize(rows)["it"]["wer"] == 0
    assert summarize(rows)["it"]["cer"] == 0
    _, json_path = write_results(tmp_path / "out", "fake", rows, environment_info())
    saved = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    assert saved["meta"]["normalizer_version"] == "1+numbers-it.1"


def test_run_stt_applies_indonesian_and_records_version(tmp_path):
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    m = tmp_path / "m.csv"
    m.write_text("id,path,reference,language,license,source\n"
                 "a,a.wav,Saya berusia dua puluh satu tahun.,id,MIT,author-written MIT test\n",
                 encoding="utf-8")
    items, problems = load_manifest(m)
    assert problems == []

    rows = run_stt(lambda path, language: "Saya berusia 21 tahun.", items, numbers="id")
    assert summarize(rows)["id"]["wer"] == 0
    assert summarize(rows)["id"]["cer"] == 0
    _, json_path = write_results(tmp_path / "out", "fake", rows, environment_info())
    saved = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    assert saved["meta"]["normalizer_version"] == "1+numbers-id.1"


def test_run_stt_does_not_infer_number_normalizer_from_manifest_language(tmp_path):
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    m = tmp_path / "m.csv"
    m.write_text("id,path,reference,language,license,source\n"
                 "a,a.wav,Bu sınavdan kırk iki puan aldım.,tr,MIT,author-written MIT test\n", encoding="utf-8")
    items, problems = load_manifest(m)
    assert problems == []

    rows = run_stt(lambda path, language: "Bu sınavdan 42 puan aldım.", items)
    assert summarize(rows)["tr"]["wer"] == pytest.approx(2 / 6)
    assert summarize(rows)["tr"]["cer"] > 0
    _, json_path = write_results(tmp_path / "out", "fake", rows, environment_info())
    saved = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    assert saved["meta"]["normalizer_version"] == "1"


@pytest.mark.parametrize("text, expected", [
    ("1,200", "1200"),
    ("12,345,678", "12345678"),
    ("3:15", "3 15"),
    ("$120", "120"),
    ("1,2", "1 2"),                                       # not a thousands separator
    ("४७२९", "4729"),                 # Devanagari digits
    ("௪௭", "47"),                               # Tamil digits
])
def test_digits(text, expected):
    assert normalize(text, numbers=True) == expected


def test_indic_text_is_untouched():
    hi = "नमस्ते दुनिया"
    assert normalize(hi, numbers=True) == normalize(hi)


def test_run_stt_passes_the_option(tmp_path):
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    m = tmp_path / "m.csv"
    m.write_text("id,path,reference,language,license,source\n"
                 "a,a.wav,order number four seven two nine,en,CC-BY-4.0,test\n", encoding="utf-8")
    items, problems = load_manifest(m)
    assert problems == []

    def engine(path, lang):
        return "order number 4729"

    assert summarize(run_stt(engine, items))["en"]["wer"] == pytest.approx(4 / 6)
    rows = run_stt(engine, items, numbers=True)
    assert summarize(rows)["en"]["wer"] == 0
    _, json_path = write_results(tmp_path / "out", "fake", rows, environment_info())
    saved = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    assert saved["meta"]["normalizer_version"] == "1+numbers-en.1"
