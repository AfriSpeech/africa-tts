# Africa TTS API server

A production HTTP + WebSocket API for Africa TTS on CPU, built on
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (ONNX Runtime) with the africa-g2p front
end. No PyTorch, no GPU. Serves **Africa TTS Turbo** by default; the unguided 24-layer bundle
works too (`AFRICA_TTS_MODEL=AfriSpeech/africa-tts`).

sherpa-onnx itself ships WebSocket/HTTP servers only for speech *recognition*; this server
fills the gap for Pocket TTS models, with the IPA front end and default voices built in.

## Run

```bash
docker build -f server/Dockerfile -t africa-tts-server .
docker run -p 8000:8000 -e AFRICA_TTS_WORKERS=4 -e AFRICA_TTS_THREADS=2 \
    -v hf-cache:/root/.cache/huggingface africa-tts-server
```

Without Docker:

```bash
pip install -r server/requirements.txt      # + apt install espeak-ng (1.51) for English
AFRICA_TTS_MODEL=AfriSpeech/africa-tts-turbo uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The bundle (`sherpa-onnx/` of the model repo) is downloaded on first start; set
`AFRICA_TTS_BUNDLE=/path/to/bundle` to use a local copy.

## API

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | | model, workers, requests in flight |
| `GET /v1/languages` | | 93 trained + 657 experimental languages (`status`, `voice_from`) |
| `POST /v1/tts` | `{"text", "lang", "voice_wav_base64"?, "format": "wav"\|"pcm"}` | audio (WAV, or raw PCM s16le 24 kHz mono); headers `X-Audio-Seconds`, `X-Compute-Seconds` |
| `POST /v1/audio/speech` | OpenAI shape: `{"input", "voice": "<language code>", "response_format"}` | audio |
| `WS /v1/tts/stream` | first message `{"text", "lang", "voice_wav_base64"?}` | `{"event":"start",...}`, binary PCM s16le chunks as they are generated, `{"event":"done"}` |

```bash
curl -X POST localhost:8000/v1/tts -H "Content-Type: application/json" \
     -d '{"text": "Akwaaba! Yɛfrɛ me Kofi.", "lang": "twi"}' -o twi.wav
python server/client.py http://localhost:8000 stream "Sannu da zuwa." --lang hau -o hausa.wav
python server/client.py http://localhost:8000 bench --concurrency 4 --requests 16
```

`lang` is an ISO 639-3 code; each language's default voice is used unless `voice_wav_base64`
carries a 3-10 s WAV to clone. Text is plain spelling (the server converts it to IPA); spell out
numbers.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AFRICA_TTS_MODEL` | `AfriSpeech/africa-tts-turbo` | HF repo whose `sherpa-onnx/` bundle to serve |
| `AFRICA_TTS_BUNDLE` | | local bundle folder (skips the download) |
| `AFRICA_TTS_WORKERS` | 2 | model instances = requests synthesised in parallel |
| `AFRICA_TTS_THREADS` | 2 | ONNX Runtime threads per instance |
| `AFRICA_TTS_MAX_QUEUE` | 32 | requests allowed to wait for a free worker before `503` |
| `AFRICA_TTS_MAX_CHARS` | 2000 | maximum text length |
| `AFRICA_TTS_API_KEY` | | if set, required as `Authorization: Bearer <key>` (HTTP and WebSocket) |

## Scaling

sherpa-onnx's Pocket TTS backend synthesises one request at a time per instance (there is no
dynamic batching for it), so throughput comes from parallel instances: size
`WORKERS x THREADS` to the CPU cores, and scale out with more containers behind a load balancer
(the server is stateless). Each instance holds its own copy of the model in memory.

Measured with the **24-layer unguided** bundle on a shared 20-vCPU server (load average ~7 from
other jobs), 4 workers x 2 threads:

| Concurrency | Throughput (seconds of audio per second) | Latency p50 (5-7 s sentences) |
|---|---|---|
| 1 | 0.59 | 8.8 s |
| 4 | 1.73 | 9.1 s |

Parallel workers scale (sherpa-onnx releases the Python GIL). Turbo does about 4x less compute
per frame, so expect roughly 4x these numbers on the same machine; a dedicated CPU will do
better than this shared one. Measure on your hardware with `client.py bench`.

## Notes

- **Streaming granularity:** sherpa-onnx emits audio per sentence (it splits on punctuation and
  merges very short sentences), so the first audio arrives once the first sentence is done.
  Shorter first sentences give earlier audio.
- **Guidance:** the guided 24-layer mode needs a second pass per frame, which sherpa-onnx does
  not run; serve it with the Python package (`AfricaTTS.load("africa-tts")`) if needed.
- **English** uses espeak-ng 1.51 (Debian bookworm in the image); the image's front end was
  checked against the training IPA on the 5,442-row parity set (all identical). espeak-ng is
  GPL-3.0-or-later.
- Models and default voices are CC-BY-NC-4.0 (non-commercial).
