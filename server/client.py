"""Test and load-test client for the Africa TTS API server.

    python server/client.py http://localhost:8000 tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o out.wav
    python server/client.py http://localhost:8000 stream "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o out.wav
    python server/client.py http://localhost:8000 bench --concurrency 4 --requests 16
"""

import argparse
import asyncio
import json
import statistics
import time
import wave
from pathlib import Path

import httpx
import websockets

SENTENCES = [
    ("twi", "Obiara wɔ ho kwan sɛ ɔtena ase asomdwoeɛ ne ahofadie mu."),
    ("hau", "Kowa na da hakkin rayuwa, da yanci, da kuma zaman lafiya."),
    ("swh", "Kila mtu ana haki ya kuishi, uhuru na usalama wa nafsi yake."),
    ("yor", "Ẹnikẹ́ni ló ní ẹ̀tọ́ sí ìwàláàyè, òmìnira àti ààbò ara rẹ̀."),
    ("eng", "Everyone has the right to life, liberty and security of person."),
]


async def tts(url: str, text: str, lang: str, out: str) -> None:
    async with httpx.AsyncClient(timeout=300) as c:
        t0 = time.monotonic()
        r = await c.post(f"{url}/v1/tts", json={"text": text, "lang": lang})
        r.raise_for_status()
        Path(out).write_bytes(r.content)
        print(f"{out}: {r.headers.get('x-audio-seconds')} s audio, "
              f"{time.monotonic() - t0:.2f} s round trip, {r.headers.get('x-compute-seconds')} s compute")


async def stream(url: str, text: str, lang: str, out: str) -> None:
    ws_url = url.replace("http", "ws", 1) + "/v1/tts/stream"
    pcm = bytearray()
    t0 = time.monotonic()
    first = None
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"text": text, "lang": lang}))
        async for msg in ws:
            if isinstance(msg, bytes):
                first = first or time.monotonic() - t0
                pcm += msg
            else:
                ev = json.loads(msg)
                if ev["event"] in ("done", "error"):
                    print(ev)
                    break
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(bytes(pcm))
    print(f"{out}: first audio after {first:.2f} s, total {time.monotonic() - t0:.2f} s, "
          f"{len(pcm) / 48000:.2f} s audio")


async def bench(url: str, concurrency: int, n: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    lat, audio_s = [], 0.0

    async def one(i: int) -> None:
        nonlocal audio_s
        lang, text = SENTENCES[i % len(SENTENCES)]
        async with sem, httpx.AsyncClient(timeout=600) as c:
            t0 = time.monotonic()
            r = await c.post(f"{url}/v1/tts", json={"text": text, "lang": lang})
            r.raise_for_status()
            lat.append(time.monotonic() - t0)
            audio_s += float(r.headers["x-audio-seconds"])

    t0 = time.monotonic()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.monotonic() - t0
    print(f"{n} requests, concurrency {concurrency}: wall {wall:.1f} s, "
          f"throughput {audio_s / wall:.2f} s of audio per second, "
          f"latency p50 {statistics.median(lat):.2f} s, max {max(lat):.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("cmd", choices=["tts", "stream", "bench"])
    ap.add_argument("text", nargs="?")
    ap.add_argument("--lang", default="twi")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "tts":
        asyncio.run(tts(a.url, a.text, a.lang, a.out))
    elif a.cmd == "stream":
        asyncio.run(stream(a.url, a.text, a.lang, a.out))
    else:
        asyncio.run(bench(a.url, a.concurrency, a.requests))


if __name__ == "__main__":
    main()
