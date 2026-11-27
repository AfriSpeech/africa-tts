# Deploy the API on a VPS

[vps/](vps/) runs the Africa TTS API on any Linux VPS (Hostinger, Contabo, Hetzner, ...) behind
[Caddy](https://caddyserver.com), which handles HTTPS. Your app calls it from its server with a
secret key; browsers never see the key.

No VPS? [huggingface/](huggingface/) runs the same API on a free Hugging Face Space (it sleeps
when unused), and [huggingface-docker/](huggingface-docker/) on a paid Docker Space. Each has
its own README.

**Size:** 4 vCPU and 8 GB RAM is comfortable (each worker holds the 440 MB model, about 1 GB in
use). 2 vCPU works with `AFRICA_TTS_WORKERS=1`, more slowly.

## Steps (Ubuntu 24.04)

1. **Domain:** add a DNS A record, e.g. `tts.baobabstack.co.zw`, pointing at the VPS's IP.
2. **Docker and firewall:**

   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
   sudo ufw allow 22 && sudo ufw allow 80 && sudo ufw allow 443 && sudo ufw --force enable
   ```

3. **Get the files and set them up:**

   ```bash
   git clone https://github.com/baobabstack-code/africa-tts.git
   cd africa-tts/deploy/vps
   cp .env.example .env
   sed -i "s/^AFRICA_TTS_API_KEY=.*/AFRICA_TTS_API_KEY=$(openssl rand -hex 24)/" .env
   nano .env            # set TTS_DOMAIN; copy the key for your app
   ```

4. **Start it:** `sudo docker compose up -d --build`. The first build takes about 10 minutes
   (PyTorch and the model); the server is ready about a minute after it starts.
5. **Check it:**

   ```bash
   curl https://tts.baobabstack.co.zw/health
   curl -X POST https://tts.baobabstack.co.zw/v1/tts \
        -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
        -d '{"text": "Akwaaba! Yɛfrɛ me Kofi.", "lang": "twi"}' -o twi.wav
   ```

   Without the key, `/v1/tts` answers `401`.
6. **Your app:** set the URL and the key (Scribe: `SPEECH_API_URL` and `SPEECH_API_KEY`).

## Running it

- Logs: `sudo docker compose logs -f tts`
- Update: `git pull && sudo docker compose up -d --build`
- Both containers restart on their own after a crash or a reboot.

## Good to know

- Requests are synthesised `AFRICA_TTS_WORKERS` at a time; others queue (up to 32), then get `503`.
- Texts are limited to `AFRICA_TTS_MAX_CHARS` characters (1,000).
- The model and voices are CC-BY-NC-4.0: non-commercial use only, with credit to AfriSpeech.
