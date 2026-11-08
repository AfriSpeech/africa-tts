# Africa TTS API server

A production HTTP + WebSocket API for Africa TTS, part of the `africa-tts` package
([africa_tts/server](../africa_tts/server)), with the IPA front end and default voices built in.
It runs on PyTorch CPU (fp32 unquantised quality) by default, or GPU with `device="cuda"`.

## Run

```bash
pip install africa-tts            # + apt install espeak-ng (1.51) for English
africa-tts serve --port 8000      # downloads the model on first start
```

With Docker:

```bash
docker build -f server/Dockerfile -t africa-tts-server .
docker run -p 8000:8000 -e AFRICA_TTS_WORKERS=2 \
    -v hf-cache:/root/.cache/huggingface africa-tts-server
```

### On Modal (serverless, scales to zero)

```bash
modal secret create africa-tts-api-key AFRICA_TTS_API_KEY=$(openssl rand -hex 24)
modal deploy server/modal_app.py
```

[`modal_app.py`](modal_app.py) builds the image (model baked in), runs 2 workers per container, scales to zero 5 minutes after the last request
and to at most 2 containers under load. It deploys in **public mode** (below), which is what powers
the [live demo Space](https://huggingface.co/spaces/AfriSpeech/africa-tts-demo).

## API

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | | model, device, workers, requests in flight |
| `GET /v1/languages` | | 93 trained + 657 experimental languages (`status`, `voice_from`) |
| `POST /v1/tts` | `{"text", "lang", "voice_wav_base64"?, "format": "wav"\|"pcm"}` | audio (WAV, or raw PCM s16le 24 kHz mono); headers `X-Audio-Seconds`, `X-Compute-Seconds` |
| `POST /v1/audio/speech` | OpenAI shape: `{"input", "voice": "<language code>", "response_format"}` | audio |
| `WS /v1/tts/stream` | first message `{"text", "lang", "voice_wav_base64"?}` | `{"event":"start",...}`, binary PCM s16le chunks as they are generated, `{"event":"done"}` |

```bash
curl -X POST localhost:8000/v1/tts -H "Content-Type: application/json" \
     -d '{"text": "Akwaaba! Yɛfrɛ me Kofi.", "lang": "twi"}' -o twi.wav
python -m africa_tts.server.client http://localhost:8000 stream "Sannu da zuwa." --lang hau -o hausa.wav
python -m africa_tts.server.client http://localhost:8000 bench --concurrency 4 --requests 16
```

`lang` is an ISO 639-3 code; each language's default voice is used unless `voice_wav_base64`
carries a 3-10 s WAV to clone. Text is plain spelling (the server converts it to IPA); spell out
numbers.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AFRICA_TTS_MODEL` | `africa-tts` | model name, HF repo or local folder (downloaded on start) |
| `AFRICA_TTS_DEVICE` | `cpu` | `cpu` (default) or `cuda` for an NVIDIA GPU |
| `AFRICA_TTS_WORKERS` | 2 | model instances = requests synthesised in parallel |
| `AFRICA_TTS_MAX_QUEUE` | 32 | requests allowed to wait for a free worker before `503` |
| `AFRICA_TTS_MAX_CHARS` | 2000 | maximum text length |
| `AFRICA_TTS_API_KEY` | | if set, required as `Authorization: Bearer <key>` (HTTP and WebSocket) |
| `AFRICA_TTS_PUBLIC` | | `1`: anonymous callers allowed under the limits below; key holders unlimited |
| `AFRICA_TTS_RATE_PER_MIN` | 6 | public mode: requests per minute per client IP |
| `AFRICA_TTS_CHARS_PER_HOUR` | 3000 | public mode: characters per hour per client IP |
| `AFRICA_TTS_CHARS_PER_DAY` | 200000 | public mode: characters per day, all anonymous callers together |
| `AFRICA_TTS_PROXY_HOPS` | 0 | trusted proxies in front (client IP from `X-Forwarded-For`); 0 on Modal, which passes the real IP |
| `AFRICA_TTS_CORS_ORIGINS` | | browser origins allowed to call the API (comma-separated, `*` for any) |

## Public mode and abuse protection

A web page cannot keep a secret: any key it sends is visible in the browser. So a page such as the
demo Space calls the API **without a key**, and the server protects itself instead:

- **Per client IP:** a few requests per minute and a character budget per hour (HTTP 429 with
  `Retry-After` when exceeded; WebSocket closes with an error event).
- **Daily budget:** total characters per day for all anonymous callers.
- **Request size:** `AFRICA_TTS_MAX_CHARS` (500 on Modal).
- **Hard cost cap:** the container limit (`max_containers`) bounds the worst case, whatever
  happens. The counters live in each container, so the effective limits scale with it.
- **CORS:** only the listed origins can call from a browser. This stops other sites embedding
  the API, but not scripts, which the limits above cover.

Your own apps use the **API key** (`Authorization: Bearer <key>`), which skips the limits; keep it
server-side.

## Scaling

Each model instance synthesises one request at a time, so throughput comes from parallel
instances: size `WORKERS` to the CPU cores, and scale out with more containers behind a
load balancer (the server is stateless). Each instance holds its own copy of the model in memory
(about 440 MB). Every instance is warmed at startup, so the first requests do not
pay for graph allocation. Measure on your hardware with `python -m africa_tts.server.client bench`.

## Notes

- **Streaming:** PyTorch streams frame by frame (80 ms chunks); the first chunk arrives
  about 0.17 s after the request on a CPU.
- **On device:** phones and embedded targets use the sherpa-onnx bundle instead
  ([mobile/](../mobile/)).
- **English** uses espeak-ng 1.51 (Debian bookworm in the image); the image's front end was
  checked against the training IPA on the 5,442-row parity set (all identical). espeak-ng is
  GPL-3.0-or-later.
- Models and default voices are CC-BY-NC-4.0 (non-commercial).
