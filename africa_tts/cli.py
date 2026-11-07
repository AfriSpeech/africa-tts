"""africa-tts command line.

  africa-tts "Akwaaba! Yɛfrɛ me Kofi." --lang twi -o out.wav
  africa-tts "Hello Ghana" --lang eng --voice me.wav -o hello.wav   # clone a 3-10 s clip
  africa-tts --ipa-only "Akwaaba!" --lang twi        # print the IPA the model would read
  africa-tts --download                              # fetch the model now (about 440 MB)
  africa-tts "Akwaaba!" --lang twi --device cuda     # on an NVIDIA GPU
  africa-tts --list-languages
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="africa-tts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?")
    ap.add_argument("--lang", "-l", help="ISO 639-3 code, e.g. twi, ewe, hau, yor, zul, eng")
    ap.add_argument("--model", "-m", default="africa-tts",
                    help="africa-tts (default), or a local folder with a downloaded copy")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda (NVIDIA GPU)")
    ap.add_argument("--voice", "-v", help="WAV to clone (default: the language's bundled voice)")
    ap.add_argument("--out", "-o", default="out.wav")
    ap.add_argument("--ipa", action="store_true", help="TEXT is already IPA")
    ap.add_argument("--ipa-only", action="store_true", help="print the IPA and exit")
    ap.add_argument("--list-languages", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="download the model on first use without asking (about 440 MB)")
    ap.add_argument("--download", action="store_true", help="download the model now and exit")
    ap.add_argument("--list-experimental", action="store_true",
                    help="untrained languages with an africa-g2p IPA table (try them and report!)")
    a = ap.parse_args(argv)

    from africa_tts import AfricaTTS, experimental_languages, languages, to_ipa

    if a.list_experimental:
        print("tier\tcode\tname\tclosest trained language\tIPA coverage")
        for code, e in experimental_languages().items():
            print(f"{e['tier']}\t{code}\t{e['name']}\t{e['nearest_name']} ({e['nearest']})\t{e['coverage']:.0%}")
        return 0
    if a.list_languages:
        for code, info in sorted(languages().items(), key=lambda kv: kv[1]["name"]):
            print(f"{code}\t{info['name']}\t{info['hours']:.1f} h")
        return 0
    if a.download:
        from africa_tts.tts import MODELS, ensure_downloaded

        ensure_downloaded(MODELS.get(a.model, a.model), download=True)
        print("Africa TTS is downloaded and cached.", file=sys.stderr)
        return 0
    if not a.text or not a.lang:
        ap.error("TEXT and --lang are required")
    if a.ipa_only:
        print(to_ipa(a.text, a.lang))
        return 0
    from africa_tts.tts import DownloadDeclined

    try:
        tts = AfricaTTS.load(a.model, download=True if a.yes else None, device=a.device)
    except DownloadDeclined as e:
        print(e, file=sys.stderr)
        return 1
    audio = tts.generate(a.text, a.lang, voice=a.voice, ipa=a.ipa)
    tts.save(audio, a.out)
    print(f"wrote {a.out} ({len(audio) / tts.sample_rate:.1f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
