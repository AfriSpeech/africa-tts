---
title: Africa TTS API
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Private text-to-speech API for 92 African languages
---

# Africa TTS API

The [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts) server by AfriSpeech on a free
Hugging Face Space. It is an API for one app, not a demo: speaking needs the secret key.

- `GET /tts-api/health` and `GET /tts-api/v1/languages` answer without a key.
- `POST /tts-api/v1/tts` needs `Authorization: Bearer <AFRICA_TTS_API_KEY>`.

Free Spaces only run Gradio apps, so [app.py](app.py) lets Gradio start the server and
attaches the ordinary API under `/tts-api`. Speech is made on the CPU. (With a paid plan, the Docker version in
`deploy/huggingface-docker` does the same without Gradio.)

## Setting it up

1. Create a Space: SDK **Gradio**, template **Blank**, the free hardware, visibility
   **Public**. A private Space asks for a Hugging Face token, which the app doesn't send;
   the secret key still keeps everyone else out.
2. In the Space's **Settings → Variables and secrets**, add a **secret** named
   `AFRICA_TTS_API_KEY` with a long random value (e.g. `openssl rand -hex 24`).
3. Upload this folder's four files to the Space: `app.py`, `requirements.txt`,
   `packages.txt` and `README.md`.
4. Wait for the build and the model download (about 10 minutes). The API's address
   is `https://<owner>-<space-name>.hf.space/tts-api`; its `/health` answers once it is ready.
5. In your app, set that address (with `/tts-api`) and the key (Scribe: `SPEECH_API_URL` and
   `SPEECH_API_KEY`).

## Good to know

- A free Space sleeps after a while without requests. The next request wakes it, which takes
  a few minutes (it downloads the 440 MB model again); until then the app says the speech
  server can't be reached.
- One request is spoken at a time and others queue.
- The model and voices are CC-BY-NC-4.0: non-commercial use only, with credit to AfriSpeech.
