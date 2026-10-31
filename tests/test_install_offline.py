"""install.sh --offline: installs from a bundle made by scripts/make_offline_bundle.sh with no git clone, no pull and no
build. Docker, curl and git are test doubles, and the bundle is made by hand, so no Docker and no network are needed."""
import gzip
import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install.sh"
FAKE_DOCKER = """#!/bin/sh
echo "$@" >> "$FAKE_LOG"
case "$1" in
  info) exit 0;;
  load) cat > /dev/null; exit 0;;
  compose)
    case "$*" in
      *"exec -T ollama ollama list"*) printf 'NAME ID\\nqwen2.5:3b abc\\nnomic-embed-text:latest def\\n'; exit 0;;
    esac; exit 0;;
esac
exit 0
"""
FAKE_CURL = "#!/bin/sh\necho '{\"status\":\"ok\"}'\nexit 0\n"
FAKE_GIT = "#!/bin/sh\necho \"git called: $@\" >> \"$FAKE_LOG\"\nexit 1\n"
THIS_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(os.uname().machine, os.uname().machine)


def _gz_tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def make_bundle(tmp, arch=THIS_ARCH, models="qwen2.5:3b,nomic-embed-text", tamper=None, evil_member=False):
    """Builds a bundle like scripts/make_offline_bundle.sh does. Returns the path of the .tar file."""
    tmp.mkdir(parents=True, exist_ok=True)
    name = "ragleap-offline-9.9.9-abc1234-" + arch
    files = {
        "repo.tar.gz": _gz_tar({"docker-compose.yml": b"services: {}\n", ".env.example": (ROOT / ".env.example").read_bytes()}),
        "images.tar.gz": gzip.compress(b"fake images"),
        "MANIFEST.txt": ("RagLeap offline bundle\nversion: 9.9.9\ncommit: abc1234\narchitecture: %s\nmodels: %s\nimages:\n"
                         % (arch, models or "none")).encode(),
        "README-OFFLINE.txt": b"read me\n",
    }
    if models:
        files["models.tar.gz"] = _gz_tar({"models/blob": b"weights"})
    sums = "".join("%s  %s\n" % (hashlib.sha256(d).hexdigest(), n) for n, d in sorted(files.items()))
    files["SHA256SUMS"] = sums.encode()
    if tamper:
        files[tamper] = files[tamper] + b"x"                     # changed after the checksums were written
    out = tmp / (name + ".tar")
    with tarfile.open(out, "w") as t:
        for n, d in files.items():
            info = tarfile.TarInfo(name + "/" + n)
            info.size = len(d)
            t.addfile(info, io.BytesIO(d))
        if evil_member:
            info = tarfile.TarInfo("../evil.txt")
            info.size = 1
            t.addfile(info, io.BytesIO(b"x"))
    Path(str(out) + ".sha256").write_text("%s  %s\n" % (hashlib.sha256(out.read_bytes()).hexdigest(), out.name))
    return out


def install(base, *args, env=None):
    base.mkdir(parents=True, exist_ok=True)
    bin_dir = base / "bin"
    bin_dir.mkdir(exist_ok=True)
    for n, body in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL), ("git", FAKE_GIT)):
        (bin_dir / n).write_text(body)
        (bin_dir / n).chmod(0o755)
    proj, log, tmpdir = base / "newproj", base / "docker.log", base / "tmp"
    log.touch()
    tmpdir.mkdir(exist_ok=True)
    e = {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "HOME": str(base), "RAGLEAP_DIR": str(proj),
         "RAGLEAP_NONINTERACTIVE": "1", "RAGLEAP_WAIT_TRIES": "2", "FAKE_LOG": str(log), "TMPDIR": str(tmpdir),
         "RAGLEAP_PORT_CHECK": "off", "RAGLEAP_NO_BROWSER": "1", "RAGLEAP_MIN_FREE_GB": "0", "RAGLEAP_MIN_FREE_GB_OLLAMA": "0"}
    e.update(env or {})
    r = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=e, stdin=subprocess.DEVNULL,
                       start_new_session=True, timeout=60)
    return r, proj, log.read_text(), tmpdir


def envfile(proj):
    d = {}
    for line in (proj / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            d[k] = v
    return d


def test_an_ollama_install_from_a_bundle_needs_no_clone_no_pull_and_no_build(tmp_path):
    bundle = make_bundle(tmp_path)
    r, proj, log, tmpdir = install(tmp_path, "--ollama", "--offline", str(bundle))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (proj / "docker-compose.yml").exists()                      # unpacked from the bundle
    e = envfile(proj)
    assert e["COMPOSE_PROJECT_NAME"] == "ragleap-core" and e["LLM_PROVIDER"] == "ollama"
    assert re.fullmatch(r"[0-9a-f]{48}", e["RAGLEAP_API_KEY"])
    assert "load" in log.split() and "git called" not in log
    assert "--build" not in log and "ollama pull" not in log and " pull qwen" not in log
    assert "compose --profile ollama up -d --pull never ollama" in log
    assert "compose --profile ollama up -d --no-build --pull never" in log
    assert "volume create --label com.docker.compose.project=ragleap-core" in log
    assert "--entrypoint tar ollama/ollama:latest -xzf /in/models.tar.gz" in log
    assert "Local models are ready" in r.stdout
    assert not list(tmpdir.glob("ragleap-offline.*"))                  # the unpacked bundle is removed


def test_the_provider_defaults_to_local_models_or_none_and_never_to_gemini(tmp_path):
    r, proj, log, _ = install(tmp_path / "a", "--offline", str(make_bundle(tmp_path / "a")))
    assert r.returncode == 0, r.stderr
    assert envfile(proj)["LLM_PROVIDER"] == "ollama" and "provider set to ollama" in r.stdout
    r, proj, log, _ = install(tmp_path / "b", "--offline", str(make_bundle(tmp_path / "b", models=None)))
    assert r.returncode == 0, r.stderr
    assert "provider set to skip" in r.stdout and "volume create" not in log and "--profile ollama" not in log
    assert "--no-build --pull never" in log


def test_gemini_is_refused_because_it_needs_the_internet(tmp_path):
    r, proj, log, _ = install(tmp_path, "--gemini", "--offline", str(make_bundle(tmp_path)))
    assert r.returncode != 0 and "needs the internet" in r.stderr
    assert not (proj / ".env").exists()


@pytest.mark.parametrize("problem", ["bad_outer", "no_checksum_file", "bad_inner", "wrong_arch", "unsafe_name", "not_a_bundle"])
def test_a_damaged_or_wrong_bundle_is_refused_before_anything_is_loaded(tmp_path, problem):
    if problem == "wrong_arch":
        bundle = make_bundle(tmp_path, arch="arm64" if THIS_ARCH == "amd64" else "amd64")
    elif problem == "bad_inner":
        bundle = make_bundle(tmp_path, tamper="images.tar.gz")
    elif problem == "unsafe_name":
        bundle = make_bundle(tmp_path, evil_member=True)
    elif problem == "not_a_bundle":
        bundle = tmp_path / "junk.tar"
        with tarfile.open(bundle, "w") as t:
            info = tarfile.TarInfo("other/file.txt")
            info.size = 1
            t.addfile(info, io.BytesIO(b"x"))
        Path(str(bundle) + ".sha256").write_text("%s  junk.tar\n" % hashlib.sha256(bundle.read_bytes()).hexdigest())
    else:
        bundle = make_bundle(tmp_path)
    if problem == "bad_outer":
        bundle.write_bytes(bundle.read_bytes() + b"tampered")
    if problem == "no_checksum_file":
        Path(str(bundle) + ".sha256").unlink()
    r, proj, log, tmpdir = install(tmp_path, "--ollama", "--offline", str(bundle))
    assert r.returncode != 0, r.stdout
    assert "load" not in log.split() and not (proj / ".env").exists() and "git called" not in log
    assert not list(tmpdir.glob("ragleap-offline.*"))


def test_a_missing_bundle_is_refused(tmp_path):
    r, proj, log, _ = install(tmp_path, "--ollama", "--offline", str(tmp_path / "nope.tar"))
    assert r.returncode != 0 and "not found" in r.stderr and "load" not in log.split()


def test_a_bundle_without_models_cannot_do_an_ollama_install(tmp_path):
    r, proj, log, _ = install(tmp_path, "--ollama", "--offline", str(make_bundle(tmp_path, models=None)))
    assert r.returncode != 0 and "--with-models" in r.stderr
    assert "--build" not in log and " up " not in log


def test_models_that_are_not_the_defaults_are_refused_with_a_clear_message(tmp_path):
    r, proj, log, _ = install(tmp_path, "--ollama", "--offline", str(make_bundle(tmp_path, models="llama3:8b")))
    assert r.returncode != 0 and "qwen2.5:3b" in r.stderr and "RAGLEAP_OLLAMA_MODEL" in r.stderr
    assert "up -d --no-build" not in log


def test_the_option_needs_a_path_and_the_help_mentions_it(tmp_path):
    r, *_ = install(tmp_path, "--offline")
    assert r.returncode == 2 and "needs the path" in r.stderr
    r, *_ = install(tmp_path, "--help")
    assert r.returncode == 0 and "--offline" in r.stdout


def test_the_bundle_path_can_come_from_the_environment(tmp_path):
    r, proj, log, _ = install(tmp_path, "--skip-ai", env={"RAGLEAP_OFFLINE_BUNDLE": str(make_bundle(tmp_path))})
    assert r.returncode == 0, r.stderr
    assert "--no-build --pull never" in log and "git called" not in log


def test_the_online_install_is_unchanged(tmp_path):
    proj = tmp_path / "newproj2"
    proj.mkdir(parents=True)
    shutil.copy(ROOT / ".env.example", proj / ".env.example")
    (proj / "docker-compose.yml").write_text("services: {}\n")
    r, _, log, _ = install(tmp_path, "--skip-ai", env={"RAGLEAP_DIR": str(proj)})
    assert r.returncode == 0, r.stderr
    assert "compose up --build -d" in log and "--no-build" not in log and "COMPOSE_PROJECT_NAME" not in (proj / ".env").read_text()
