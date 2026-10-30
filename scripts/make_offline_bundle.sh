#!/usr/bin/env bash
# Builds an offline bundle of RagLeap: one archive that holds the code, every Docker image the compose file needs
# (built here, so no internet is needed to build later) and, optionally, Ollama models. Run it on a machine WITH
# internet, carry the archive to the machine WITHOUT internet. See the README-OFFLINE.txt inside the archive.
#
# Usage: bash scripts/make_offline_bundle.sh [--out DIR] [--ref GIT_REF] [--with-models "name,name"]
#   --out DIR            where to write the archive (default: ./ragleap-offline)
#   --ref GIT_REF        which commit or tag to bundle (default: HEAD of this checkout)
#   --with-models LIST   Ollama models to include, for example "llama3.2:3b,nomic-embed-text" (large; check their licences)
# Environment: MIN_FREE_GB (default 15) is the free disk space required in the output and temp folders.
# The bundle is for the CPU type of the machine that builds it (amd64 or arm64). It is checked with SHA-256 only, not signed.
set -euo pipefail

OUT="./ragleap-offline"; REF="HEAD"; MODELS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="${2:?--out needs a folder}"; shift 2 ;;
    --ref) REF="${2:?--ref needs a git ref}"; shift 2 ;;
    --with-models) MODELS="${2:?--with-models needs a comma separated list}"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "STOP: unknown option $1 (try --help)" >&2; exit 2 ;;
  esac
done
say() { echo "==> $*"; }
die() { echo "STOP: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
docker compose version >/dev/null 2>&1 || die "the docker compose plugin was not found"
for t in git tar gzip; do command -v "$t" >/dev/null 2>&1 || die "$t is not installed"; done
if command -v sha256sum >/dev/null 2>&1; then SHA() { sha256sum "$@"; }; elif command -v shasum >/dev/null 2>&1; then SHA() { shasum -a 256 "$@"; }; else die "sha256sum or shasum is needed"; fi

if [ -n "$MODELS" ]; then
  IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
  for m in "${MODEL_LIST[@]}"; do
    [[ "$m" =~ ^[A-Za-z0-9._:/-]{1,100}$ ]] || die "model name '$m' has characters that are not allowed"
  done
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git -C "$HERE" rev-parse --git-dir >/dev/null 2>&1 || die "run this from a git checkout of ragleap-core"
COMMIT="$(git -C "$HERE" rev-parse --short "$REF" 2>/dev/null)" || die "git ref '$REF' not found"
VERSION="$(git -C "$HERE" show "$REF:core/api.py" 2>/dev/null | sed -n 's/^ *version="\([^"]*\)",.*/\1/p' | head -1)"
[ -n "$VERSION" ] || VERSION="unknown"
case "$(uname -m)" in x86_64|amd64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;; *) ARCH="$(uname -m)" ;; esac
NAME="ragleap-offline-${VERSION}-${COMMIT}-${ARCH}"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
[ -e "$OUT/$NAME.tar" ] && die "$OUT/$NAME.tar already exists; remove it or choose another --out"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ragleap-bundle.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
MIN_FREE_GB="${MIN_FREE_GB:-15}"
for d in "$OUT" "$WORK"; do
  free_kb="$(df -Pk "$d" | awk 'NR==2{print $4}')"
  [ "$free_kb" -ge $((MIN_FREE_GB * 1048576)) ] || die "less than ${MIN_FREE_GB} GB free in $d"
done

say "Bundling $VERSION ($COMMIT, $ARCH)"
PROJECT=ragleap-core
SRC="$WORK/$PROJECT"; mkdir -p "$SRC" "$WORK/pack"
git -C "$HERE" archive --format=tar "$REF" | tar -x -C "$SRC"
: > "$SRC/.env"                      # compose needs the file to exist; it stays empty and is not bundled
git -C "$HERE" archive --format=tar.gz -o "$WORK/pack/repo.tar.gz" "$REF"
CP=(docker compose -p "$PROJECT" --profile full --profile sandbox --profile ollama)

say "Building the images (this needs internet and takes a few minutes)"
( cd "$SRC" && "${CP[@]}" build )
say "Pulling the other images"
( cd "$SRC" && { "${CP[@]}" pull --ignore-buildable || "${CP[@]}" pull --ignore-pull-failures; } )
docker pull python:3.11-slim >/dev/null
IMAGES="$( cd "$SRC" && "${CP[@]}" config --images | sort -u )"
IMAGES="$(printf '%s\npython:3.11-slim\n' "$IMAGES" | grep -v '^$' | sort -u)"
[ -n "$IMAGES" ] || die "compose reported no images"

say "Saving the images (large)"
# shellcheck disable=SC2086
docker save $IMAGES | gzip > "$WORK/pack/images.tar.gz"

if [ -n "$MODELS" ]; then
  say "Downloading Ollama models: $MODELS"
  mkdir -p "$WORK/ollama"
  docker run --rm -v "$WORK/ollama:/root/.ollama" --entrypoint sh ollama/ollama:latest -c \
    'ollama serve >/dev/null 2>&1 & sleep 6; for m in "$@"; do ollama pull "$m" || exit 1; done' sh "${MODEL_LIST[@]}"
  tar -czf "$WORK/pack/models.tar.gz" -C "$WORK/ollama" .
fi

{
  echo "RagLeap offline bundle"
  echo "version: $VERSION"
  echo "commit: $COMMIT"
  echo "architecture: $ARCH"
  echo "built (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "models: ${MODELS:-none}"
  echo "images:"
  for i in $IMAGES; do echo "  $i $(docker image inspect --format '{{.Id}}' "$i" 2>/dev/null || echo unknown)"; done
} > "$WORK/pack/MANIFEST.txt"

cat > "$WORK/pack/README-OFFLINE.txt" <<'TXT'
RagLeap offline bundle. Use it on a machine with Docker and the docker compose plugin and NO internet.
It only runs on the CPU type named in MANIFEST.txt. Hosted model providers (Gemini and similar) cannot work offline;
use local models (Ollama) and local embeddings.

1. Check the files:      sha256sum -c SHA256SUMS            (on macOS: shasum -a 256 -c SHA256SUMS)
2. Unpack the code:      mkdir ragleap-core && tar -xzf repo.tar.gz -C ragleap-core
3. Load the images:      gunzip -c images.tar.gz | docker load
4. Models (if bundled):  docker volume create --label com.docker.compose.project=ragleap-core \
                           --label com.docker.compose.volume=ragleap_core_ollama ragleap-core_ragleap_core_ollama
                         docker run --rm -v ragleap-core_ragleap_core_ollama:/root/.ollama -v "$PWD":/in:ro \
                           --entrypoint tar ollama/ollama:latest -xzf /in/models.tar.gz -C /root/.ollama
5. Configure and start:  cd ragleap-core && cp .env.example .env     (edit .env: choose your own keys and passwords)
                         docker compose -p ragleap-core up -d --no-build --pull never
   Add --profile ollama (local models), --profile graph, --profile voice, --profile queue or --profile sandbox as needed.

This procedure has been written but not yet tried on an air-gapped machine. Tell the maintainers what you find.
TXT
( cd "$WORK/pack" && SHA $(ls | grep -v '^SHA256SUMS$' | sort) > SHA256SUMS )

say "Making the archive"
mv "$WORK/pack" "$WORK/$NAME"
tar -cf "$OUT/$NAME.tar" -C "$WORK" "$NAME"
( cd "$OUT" && SHA "$NAME.tar" > "$NAME.tar.sha256" && SHA -c "$NAME.tar.sha256" >/dev/null ) || die "the checksum of the archive does not verify"
say "Done: $OUT/$NAME.tar ($(du -h "$OUT/$NAME.tar" | cut -f1)) and $NAME.tar.sha256"
