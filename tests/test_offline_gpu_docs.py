"""The GPU compose override and docs/offline-and-gpu.md: the override only adds a GPU reservation to the Ollama service, and the
guide stays in step with the scripts it names (paths, flags, default models)."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GUIDE = (ROOT / "docs" / "offline-and-gpu.md").read_text()
OVERRIDE = yaml.safe_load((ROOT / "docker-compose.gpu.yml").read_text())
BASE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def test_the_override_only_adds_a_gpu_reservation_to_the_ollama_service():
    assert list(OVERRIDE) == ["services"] and list(OVERRIDE["services"]) == ["ollama"]
    ollama = OVERRIDE["services"]["ollama"]
    assert list(ollama) == ["deploy"]
    assert ollama["deploy"]["resources"]["reservations"]["devices"] == [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]


def test_merged_with_the_base_file_the_ollama_service_keeps_everything_else():
    assert "deploy" not in BASE["services"]["ollama"]
    merged = _merge(BASE, OVERRIDE)["services"]["ollama"]
    for key in ("image", "profiles", "volumes", "environment", "restart"):
        assert merged[key] == BASE["services"]["ollama"][key]
    assert "ports" not in merged and "privileged" not in merged                 # still no published port
    assert merged["deploy"]["resources"]["reservations"]["devices"][0]["driver"] == "nvidia"


def test_the_guide_names_files_that_exist():
    for path in set(re.findall(r"(?<![\w/.-])((?:scripts/[\w./-]+|docker-compose[\w.-]*\.yml|install\.sh))", GUIDE)):
        assert (ROOT / path).exists(), path
    assert "docker-compose.gpu.yml" in GUIDE and "make_offline_bundle.sh" in GUIDE


def test_the_flags_in_the_guide_exist_in_the_scripts():
    assert "--with-models" in (ROOT / "scripts" / "make_offline_bundle.sh").read_text() and "--with-models" in GUIDE
    installer = (ROOT / "install.sh").read_text()
    assert "--offline" in installer and "--offline" in GUIDE
    assert "--ollama" in installer and "--ollama" in GUIDE


def test_the_default_models_in_the_guide_are_the_installers_defaults():
    installer = (ROOT / "install.sh").read_text()
    chat = re.search(r'CHAT_MODEL="\$\{RAGLEAP_OLLAMA_MODEL:-([^}]+)\}"', installer).group(1)
    embed = re.search(r'EMBED_MODEL="([^"]+)"', installer).group(1)
    assert chat in GUIDE and embed in GUIDE and "RAGLEAP_OLLAMA_MODEL" in GUIDE


def test_the_compose_file_setting_in_the_guide_matches_the_override_name():
    assert "COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml" in GUIDE
    assert "-f docker-compose.yml -f docker-compose.gpu.yml --profile ollama up -d" in GUIDE


def test_the_guide_is_honest_about_what_is_untested_and_unsigned():
    lowered = GUIDE.lower()
    for needle in ("not yet tried", "not signed", "no speed claims", "hosted ai providers"):
        assert needle in lowered, needle


def test_the_readme_links_to_the_guide():
    readme = (ROOT / "README.md").read_text()
    assert "(docs/offline-and-gpu.md)" in readme
