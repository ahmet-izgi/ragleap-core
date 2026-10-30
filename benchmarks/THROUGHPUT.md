# Throughput checks

`http_throughput.py` measures requests per second and latency percentiles of a running RagLeap app. It uses only the
Python standard library. It is a measurement tool: no throughput number is claimed for RagLeap until it has been
measured on the hardware you care about, and any number you quote must come with the machine, the model settings and
`WEB_CONCURRENCY`.

## Run it

```bash
read -rs RAGLEAP_PERSONAL_KEY; export RAGLEAP_PERSONAL_KEY      # nothing is shown; paste the key, press Enter
python3 benchmarks/http_throughput.py --url http://127.0.0.1:8000 --paths /health,/me \
    --key-env RAGLEAP_PERSONAL_KEY --concurrency 8 --seconds 10
```

`/health` needs no key and shows the cost of the web server alone. `/me` needs a key and also shows the cost of
checking it. Use `--json` for machine-readable output. The tool limits itself to 64 threads and 300 seconds. It exits
with 1 if any request failed. Do not point it at a server other people are using.

It only sends GET requests to the paths you name. It does not call the model, so it says nothing about chat speed,
which depends on the model provider and the model.

## Web workers

The app starts one web process by default. Uvicorn reads `WEB_CONCURRENCY` from the environment, so setting it in
`.env` (for example `WEB_CONCURRENCY=2`) starts that many processes. More processes only help when the machine has
spare CPU cores and memory; on a 1 CPU server they do not. Measure before and after.

State that is per process, so it behaves differently with more than one process:

| State | With N processes |
|---|---|
| Failed-login throttle | Each process counts alone, so the limit is up to N times higher |
| `/metrics` counters | Each scrape reaches one process, so the counts are partial |
| Key and owner lookups | Cached for 30 seconds in each process |
| Scheduled jobs | Safe: one process holds a database lock and runs them |

## Queue workers

Queue workers (`docker compose --profile queue up -d --scale worker=3`) only run integration syncs, and only when
`REDIS_URL` is set. Without `REDIS_URL` the app runs syncs itself and the worker idles.
