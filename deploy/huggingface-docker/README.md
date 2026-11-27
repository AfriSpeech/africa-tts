---
title: Africa TTS API
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: Private text-to-speech API for 92 African languages
---

# Africa TTS API

The [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts) server by AfriSpeech, running
on CPU. It is an API for one app, not a demo: every request needs the secret key.

- `GET /health` and `GET /v1/languages` answer without a key.
- `POST /v1/tts` needs `Authorization: Bearer <AFRICA_TTS_API_KEY>`.

## Setting it up

1. Create a Space with the **Docker** SDK and the free **CPU basic** hardware. Keep it
   **public**: a private Space asks for a Hugging Face token, which the app doesn't send.
   The secret key still keeps everyone else out.
2. Upload this folder's `Dockerfile` and `README.md` to the Space.
3. In the Space's **Settings → Variables and secrets**, add a **secret** named
   `AFRICA_TTS_API_KEY` with a long random value (e.g. `openssl rand -hex 24`).
4. Wait for the build (about 10 minutes). The address is
   `https://<your-username>-<space-name>.hf.space`.
5. In your app, set that address and the key (Scribe: `SPEECH_API_URL` and `SPEECH_API_KEY`).

## Good to know

- A free Space sleeps after 48 hours without requests. The next request wakes it, which takes
  a minute or two; until then the app says the speech server can't be reached.
- The free hardware has 2 vCPUs, so one request is spoken at a time and others queue.
- The model and voices are CC-BY-NC-4.0: non-commercial use only, with credit to AfriSpeech.
