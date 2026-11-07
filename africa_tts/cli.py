"""africa-tts command line.

  africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o out.wav
  africa-tts "Hello Ghana" --lang eng --voice me.wav -o hello.wav   # clone a 3-10 s clip
  africa-tts --ipa-only "Akwaaba!" --lang twi        # print the IPA the model would read
  africa-tts --download                              # fetch the model now
  africa-tts --list-languages
  africa-tts serve                                   # HTTP + WebSocket API on port 8000

Engines: audio.cpp on the CPU (default, no PyTorch), or PyTorch with
pip install "africa-tts[torch]":
  africa-tts "Akwaaba!" --lang twi --backend torch   # unquantised, CPU
  africa-tts "Akwaaba!" --lang twi --device cuda     # NVIDIA GPU
"""

import argparse
import os
import sys


def _engine_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", "-m", default="africa-tts",
                    help="africa-tts (default), or a local folder with a downloaded copy")
    ap.add_argument("--backend", choices=["audiocpp", "torch"], default=None,
                    help="audiocpp (default: CPU, no PyTorch) or torch (needs the torch extra)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda (GPU; implies torch)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (audio.cpp)")


def serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="africa-tts serve",
                                 description="Run the Africa TTS API server (HTTP + WebSocket).")
    _engine_args(ap)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--workers", type=int, default=None, help="parallel requests (default 2)")
    a = ap.parse_args(argv)
    env = {"AFRICA_TTS_MODEL": a.model, "AFRICA_TTS_DEVICE": a.device,
           "AFRICA_TTS_BACKEND": a.backend or ("torch" if a.device != "cpu" else "audiocpp")}
    if a.threads:
        env["AFRICA_TTS_THREADS"] = str(a.threads)
    if a.workers:
        env["AFRICA_TTS_WORKERS"] = str(a.workers)
    os.environ.update(env)
    import uvicorn

    uvicorn.run("africa_tts.server.app:app", host=a.host, port=a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        return serve(argv[1:])
    ap = argparse.ArgumentParser(prog="africa-tts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?")
    ap.add_argument("--lang", "-l", help="ISO 639-3 code, e.g. twi, ewe, hau, yor, zul, eng")
    _engine_args(ap)
    ap.add_argument("--voice", "-v", help="WAV to clone (default: the language's bundled voice)")
    ap.add_argument("--out", "-o", default="out.wav")
    ap.add_argument("--ipa", action="store_true", help="TEXT is already IPA")
    ap.add_argument("--ipa-only", action="store_true", help="print the IPA and exit")
    ap.add_argument("--list-languages", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="download the model on first use without asking")
    ap.add_argument("--download", action="store_true", help="download the model now and exit")
    ap.add_argument("--list-experimental", action="store_true",
                    help="untrained languages with an africa-g2p IPA table (try them and report!)")
    a = ap.parse_args(argv)

    from africa_tts import experimental_languages, languages, to_ipa

    if a.list_experimental:
        print("tier\tcode\tname\tclosest trained language\tIPA coverage")
        for code, e in experimental_languages().items():
            print(f"{e['tier']}\t{code}\t{e['name']}\t{e['nearest_name']} ({e['nearest']})\t{e['coverage']:.0%}")
        return 0
    if a.list_languages:
        for code, info in sorted(languages().items(), key=lambda kv: kv[1]["name"]):
            print(f"{code}\t{info['name']}\t{info['hours']:.1f} h")
        return 0
    backend = a.backend or ("torch" if a.device != "cpu" else "audiocpp")
    from africa_tts.tts import AfricaTTS, DownloadDeclined, model_dir

    if a.download:
        model_dir(a.model, backend, download=True)
        print("Africa TTS is downloaded and cached.", file=sys.stderr)
        return 0
    if not a.text or not a.lang:
        ap.error("TEXT and --lang are required")
    if a.ipa_only:
        print(to_ipa(a.text, a.lang))
        return 0
    try:
        tts = AfricaTTS.load(a.model, backend=backend, device=a.device, threads=a.threads,
                             download=True if a.yes else None)
    except (DownloadDeclined, ImportError, ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1
    audio = tts.generate(a.text, a.lang, voice=a.voice, ipa=a.ipa)
    tts.save(audio, a.out)
    print(f"wrote {a.out} ({len(audio) / tts.sample_rate:.1f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
