"""scripts/make_offline_bundle.sh with a test double for Docker (no Docker, no network needed). The script is copied into a
tiny temporary git repository so the tests stay fast."""
import hashlib
import io
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash") or "bash"
SCRIPT = ROOT / "scripts" / "make_offline_bundle.sh"
DOCKER_DOUBLE = r'''#!/usr/bin/env bash
echo "$*" >> "${STUB_LOG:-/dev/null}"
case "$1" in
  compose)
    case "$*" in
      *" version"*) exit 0;;
      *"config --images"*) printf 'ragleap-core-app\nragleap-core-worker\nragleap-core-sandbox\npgvector/pgvector:pg16\nredis:7-alpine\nollama/ollama:latest\n'; exit 0;;
      *" pull "*) [ -n "${STUB_PULL_FAIL_FIRST:-}" ] && [[ "$*" == *"--ignore-buildable"* ]] && exit 1; exit 0;;
      *) exit 0;;
    esac;;
  pull) exit 0;;
  save) printf 'fake image data\n'; exit 0;;
  image) echo "sha256:0123456789abcdef"; exit 0;;
  run)
    d=""; for a in "$@"; do case "$a" in *:/root/.ollama) d="${a%%:*}";; esac; done
    [ -n "$d" ] && { mkdir -p "$d/models"; echo weights > "$d/models/blob"; }; exit 0;;
  *) exit 0;;
esac
'''

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash")


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "core").mkdir()
    shutil.copy(SCRIPT, repo / "scripts" / "make_offline_bundle.sh")
    (repo / "core" / "api.py").write_text('app = FastAPI(\n    version="9.9.9",\n)\n')
    (repo / "docker-compose.yml").write_text("services: {}\n")
    (repo / ".env.example").write_text("EXAMPLE=1\n")
    git_env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo)] + cmd, check=True, env=git_env)
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "docker").write_text(DOCKER_DOUBLE)
    (stub / "docker").chmod((stub / "docker").stat().st_mode | stat.S_IXUSR)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    return {"repo": repo, "out": tmp_path / "out", "tmp": tmp, "log": tmp_path / "docker.log",
            "env": dict(os.environ, PATH=str(stub) + os.pathsep + os.environ["PATH"], MIN_FREE_GB="0",
                        STUB_LOG=str(tmp_path / "docker.log"), TMPDIR=str(tmp))}


def run(e, *args, extra_env=None, path=None):
    environ = dict(e["env"], **(extra_env or {}))
    if path is not None:
        environ["PATH"] = path
    return subprocess.run([BASH, str(e["repo"] / "scripts" / "make_offline_bundle.sh"), "--out", str(e["out"]), *args],
                          capture_output=True, text=True, env=environ, cwd=str(e["repo"]))


def archive(e):
    found = sorted(e["out"].glob("ragleap-offline-*.tar"))
    assert len(found) == 1, found
    return found[0]


def members(path):
    with tarfile.open(path) as t:
        return {m.name.split("/", 1)[1]: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}


def test_it_builds_one_archive_whose_checksums_verify(env):
    r = run(env)
    assert r.returncode == 0, r.stderr
    tar_path = archive(env)
    assert "-9.9.9-" in tar_path.name
    line = Path(str(tar_path) + ".sha256").read_text().split()
    assert line[0] == hashlib.sha256(tar_path.read_bytes()).hexdigest() and line[1] == tar_path.name
    files = members(tar_path)
    assert set(files) == {"repo.tar.gz", "images.tar.gz", "MANIFEST.txt", "SHA256SUMS", "README-OFFLINE.txt"}
    for sums_line in files["SHA256SUMS"].decode().strip().splitlines():
        digest, name = sums_line.split()
        assert hashlib.sha256(files[name]).hexdigest() == digest, name
    assert "SHA256SUMS" not in files["SHA256SUMS"].decode()


def test_the_manifest_and_the_code_archive_hold_what_they_should(env):
    assert run(env).returncode == 0
    files = members(archive(env))
    manifest = files["MANIFEST.txt"].decode()
    assert "version: 9.9.9" in manifest and "models: none" in manifest and "architecture:" in manifest
    for image in ("ragleap-core-app", "pgvector/pgvector:pg16", "python:3.11-slim", "ollama/ollama:latest"):
        assert image in manifest
    import gzip
    assert gzip.decompress(files["images.tar.gz"]) == b"fake image data\n"
    with tarfile.open(fileobj=io.BytesIO(files["repo.tar.gz"])) as t:
        names = t.getnames()
    assert "docker-compose.yml" in names and ".env" not in names


def test_models_are_added_only_when_asked_for(env):
    assert run(env, "--with-models", "llama3.2:3b,nomic-embed-text").returncode == 0
    files = members(archive(env))
    assert "models.tar.gz" in files and "models: llama3.2:3b,nomic-embed-text" in files["MANIFEST.txt"].decode()
    with tarfile.open(fileobj=io.BytesIO(files["models.tar.gz"])) as t:
        assert any(n.endswith("models/blob") for n in t.getnames())


def test_a_model_name_with_odd_characters_is_refused_before_anything_runs(env):
    r = run(env, "--with-models", "ok-model;touch pwned")
    assert r.returncode != 0 and "not allowed" in r.stderr
    assert not list(env["out"].glob("*.tar"))
    assert not env["log"].exists() or "build" not in env["log"].read_text()


def test_it_uses_the_project_name_and_profiles_the_installer_expects(env):
    assert run(env).returncode == 0
    log = env["log"].read_text()
    assert "compose -p ragleap-core --profile full --profile sandbox --profile ollama build" in log
    assert "save " in log and "pull python:3.11-slim" in log


def test_an_older_compose_without_ignore_buildable_still_works(env):
    r = run(env, extra_env={"STUB_PULL_FAIL_FIRST": "1"})
    assert r.returncode == 0, r.stderr
    assert "--ignore-pull-failures" in env["log"].read_text()


def test_missing_docker_and_low_disk_stop_with_a_clear_message(env):
    r = run(env, path="/nonexistent")
    assert r.returncode == 1 and "docker is not installed" in r.stderr
    r = run(env, extra_env={"MIN_FREE_GB": "99999999"})
    assert r.returncode == 1 and "less than" in r.stderr and not list(env["out"].glob("*.tar"))


def test_it_never_overwrites_an_existing_archive(env):
    assert run(env).returncode == 0
    first = archive(env).read_bytes()
    r = run(env)
    assert r.returncode == 1 and "already exists" in r.stderr
    assert archive(env).read_bytes() == first


def test_it_leaves_no_temporary_files_and_does_not_touch_the_checkout(env):
    assert run(env).returncode == 0
    assert not list(env["tmp"].glob("ragleap-bundle.*"))
    assert subprocess.run(["git", "-C", str(env["repo"]), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_help_works_and_unknown_options_are_refused(env):
    assert "--with-models" in run(env, "--help").stdout
    r = run(env, "--nonsense")
    assert r.returncode == 2 and "unknown option" in r.stderr
