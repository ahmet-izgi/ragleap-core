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


def test_turkish_number_words_and_metrics():
    # Author-written examples checked against TDK spelling guidance:
    # https://tdk.gov.tr/icerik/yazim-kurallari/sayilarin-yazilisi/
    # No external corpus or number-word list is used.
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
    # TDK's examples include üçüncü, beşinci, and ordinal suffixes on digits.
    # The other composed examples are author-written and checked for Turkish usage.
    examples = [
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
