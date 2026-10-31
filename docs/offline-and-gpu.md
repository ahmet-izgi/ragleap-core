# Offline install and GPU

For computers with no internet (banks, research institutes, secure labs) and for machines with an NVIDIA GPU.
Everything here is **written but not yet tried on a real air-gapped machine or a real GPU**. Please tell us what you
find (see the testing issues linked from the README, section "Tested by others").

## What works with no internet

- The whole stack in Docker: the app, PostgreSQL, Redis, and optionally Neo4j, the worker and the sandbox.
- Local models through Ollama: chat and embeddings.
- **Not available offline:** hosted AI providers (Gemini, OpenAI and the others), hosted speech services, and anything
  that calls a web page. The installer refuses `--gemini` in offline mode.

## 1. Build the bundle (on a computer WITH internet)

You need Docker with the compose plugin, Git, and a checkout of this repository. The bundle only works on the CPU
type of the computer that builds it (amd64 or arm64), so build on the same kind of machine you will install on.

```bash
bash scripts/make_offline_bundle.sh --out ./bundle --with-models "qwen2.5:3b,nomic-embed-text"
```

- It builds the images, pulls the others, and writes one `.tar` file with a `.tar.sha256` next to it.
- The installer expects these two models by default (`qwen2.5:3b` for chat, `nomic-embed-text` for embeddings).
  Other chat models work if you set `RAGLEAP_OLLAMA_MODEL` when installing, but `nomic-embed-text` must be in the bundle.
- It needs about 15 GB of free disk space and downloads several GB. Models are large and have their own licences.
- The bundle is checked with SHA-256 only. It is **not signed**, so keep control of how it travels.

## 2. Carry it

Copy to the offline computer: the `.tar` file, the `.tar.sha256` file (same folder), and `install.sh` from the same
checkout. The offline computer needs Docker with the compose plugin, and `curl`. It does not need Git.

## 3. Install

```bash
bash install.sh --offline /path/to/ragleap-offline-....tar --ollama
```

Before anything is loaded, the installer checks the file against its `.sha256`, the names inside the archive, the
checksums of every file inside, and that the bundle's CPU type matches this computer. Then it loads the images, unpacks
the code and the models, and starts with no build and no pull. Without `--ollama` it picks local models if the bundle
has them, and otherwise skips the AI choice so you can set it later in the dashboard.

Limits: an existing install folder is used as it is (its compose file must match the bundle), and there is no upgrade
procedure for offline installs yet: for now, build a new bundle and install into a new folder.

## GPU (NVIDIA)

The bundled Ollama runs on the CPU by default. On a Linux host with an NVIDIA GPU, install the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then start with the override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile ollama up -d
```

To make it permanent, add `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` to `.env` (use `;` instead of `:` on
Windows). To check that the GPU is used, run `docker compose --profile ollama exec ollama nvidia-smi` and look at the Ollama log for a
line about the GPU. If the model does not fit in the GPU memory, Ollama falls back to the CPU for part or all of it.

We make no speed claims: results depend on the GPU, the model and the settings. Windows with WSL 2 and macOS are not
covered here (Docker on macOS cannot pass a GPU to containers).

## What is not covered yet

- A tried-and-confirmed air-gapped install on real hardware, arm64 bundles, and a signed bundle.
- Voice models: `ragleap-voice` is still pre-alpha. When it is complete its models will be added to the bundle.
- Upgrades of an offline install, and the Windows wrapper `install.ps1`.
