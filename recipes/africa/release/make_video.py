"""Slideshow video of the samples: one slide per language (name, country, the sentence, the
English meaning, a waveform with a moving playhead) while its audio plays.

Usage: python make_video.py SAMPLES_DIR OUT.mp4 [--label turbo] [--article 3] [--countries c.json]
  SAMPLES_DIR: meta.json + audio/ from synth_samples.py. Needs Pillow, numpy, soundfile, ffmpeg.
"""

import argparse
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

W = H = 1080
BG, FG, MUTED, ACCENT, LINE = "#FBFAF7", "#1B1A17", "#6B675F", "#A0471D", "#E7E3DA"
FONTS = Path("/usr/share/fonts/truetype/noto")
ART_EN = {"1": "All human beings are born free and equal in dignity and rights.",
          "3": "Everyone has the right to life, liberty and security of person.",
          "5": "No one shall be subjected to torture or to cruel, inhuman or degrading treatment."}
NICE = {"english_eng": "English", "swahili_swa": "Swahili", "twi_twi": "Twi", "yoruba_yor": "Yoruba",
        "xhosa_xho": "Xhosa", "tigrinya_tir": "Tigrinya", "Moore": "Mooré"}
COUNTRY_OVERRIDE = {
    "eng": "Ghanaian English", "swh": "Tanzania · Kenya · Uganda · DR Congo",
    "hau": "Nigeria · Niger · Ghana", "yor": "Nigeria · Benin", "ewe": "Ghana · Togo",
    "tir": "Eritrea · Ethiopia", "lin": "DR Congo · Congo", "nya": "Malawi · Zambia · Mozambique",
    "wol": "Senegal · The Gambia", "kin": "Rwanda", "run": "Burundi", "fat": "Ghana",
    "twi": "Ghana", "gaa": "Ghana", "zul": "South Africa", "xho": "South Africa",
    "sna": "Zimbabwe", "amh": "Ethiopia", "ibo": "Nigeria", "fon": "Benin", "kab": "Algeria",
    "mlg": "Madagascar", "sag": "Central African Republic", "kik": "Kenya", "lug": "Uganda",
}
CC = {"GH": "Ghana", "NG": "Nigeria", "TZ": "Tanzania", "KE": "Kenya", "ET": "Ethiopia",
      "CD": "DR Congo", "CG": "Congo", "BJ": "Benin", "TG": "Togo", "CI": "Côte d'Ivoire",
      "CM": "Cameroon", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe", "MZ": "Mozambique",
      "MW": "Malawi", "AO": "Angola", "NA": "Namibia", "UG": "Uganda", "RW": "Rwanda",
      "BI": "Burundi", "SN": "Senegal", "LR": "Liberia", "SL": "Sierra Leone", "GN": "Guinea",
      "BF": "Burkina Faso", "ML": "Mali", "CV": "Cabo Verde", "GW": "Guinea-Bissau",
      "MU": "Mauritius", "MG": "Madagascar", "CF": "Central African Republic", "SS": "South Sudan",
      "TD": "Chad", "DZ": "Algeria", "SZ": "Eswatini", "LS": "Lesotho", "BW": "Botswana",
      "ER": "Eritrea", "GA": "Gabon", "NE": "Niger"}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / {
        "serif-bold": "NotoSerif-Bold.ttf", "sans": "NotoSans-Regular.ttf",
        "sans-bold": "NotoSans-Bold.ttf", "italic": "NotoSerif-Italic.ttf",
        "ethiopic": "NotoSansEthiopic-Regular.ttf"}[kind]), size)


def is_ethiopic(s: str) -> bool:
    return any(0x1200 <= ord(c) <= 0x139F or 0x2D80 <= ord(c) <= 0x2DDF for c in s)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = []
    for w in text.replace("፡", "፡ ").split():  # Ge'ez word separator: allow a break after it
        while draw.textlength(w, font=f) > width:  # still too long: break by characters
            k = len(w)
            while k > 1 and draw.textlength(w[:k], font=f) > width:
                k -= 1
            words.append(w[:k])
            w = w[k:]
        words.append(w)
    lines, cur = [], ""
    for w in words:
        t = (cur + w if cur.endswith("፡") else f"{cur} {w}").strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def base(draw: ImageDraw.ImageDraw, counter: str = "") -> None:
    draw.text((72, 64), "AFRICA TTS", font=font("sans-bold", 26), fill=ACCENT)
    if counter:
        f = font("sans", 26)
        draw.text((W - 72 - draw.textlength(counter, font=f), 64), counter, font=f, fill=MUTED)
    draw.line([(72, 112), (W - 72, 112)], fill=LINE, width=2)


def waveform(audio: np.ndarray, width: int, height: int) -> list[float]:
    n = width // 6
    chunks = np.array_split(np.abs(audio), n)
    env = np.array([c.max() if len(c) else 0 for c in chunks])
    env = env / (env.max() + 1e-9)
    return list(np.maximum(env, 0.04) * height)


def lang_slide(path: Path, name: str, country: str, text: str, gloss: str, audio: np.ndarray,
               counter: str) -> tuple[int, int, int]:
    """Returns the waveform box (x0, y_mid, width) so ffmpeg can draw the playhead."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    base(d, counter)
    y = 190
    d.text((72, y), name, font=font("serif-bold", 92), fill=FG)
    y += 130
    d.text((76, y), country, font=font("sans", 34), fill=MUTED)
    y += 90
    tf = font("ethiopic" if is_ethiopic(text) else "sans", 44)
    lines = wrap(d, text, tf, W - 144)
    if len(lines) > 5:
        tf = font("ethiopic" if is_ethiopic(text) else "sans", 36)
        lines = wrap(d, text, tf, W - 144)[:6]
    for line in lines:
        d.text((72, y), line, font=tf, fill=FG)
        y += int(tf.size * 1.4)
    y += 16
    gf = font("italic", 30)
    for line in wrap(d, f"“{gloss}”", gf, W - 144):
        d.text((72, y), line, font=gf, fill=MUTED)
        y += 44
    bars = waveform(audio, W - 144, 120)
    draw_bars(d, bars, 0.0)
    d.text((72, H - 80), "Synthesised speech · voice cloned from a short clip of the language",
           font=font("sans", 22), fill=MUTED)
    img.save(path)
    return bars


WX0, WMID = 72, 905


def draw_bars(d: ImageDraw.ImageDraw, bars: list[float], progress: float) -> None:
    n = len(bars)
    for i, h in enumerate(bars):
        x = WX0 + i * 6
        played = i < progress * n
        d.rounded_rectangle([x, WMID - h / 2, x + 3, WMID + h / 2], radius=1,
                            fill=ACCENT if played else "#D9C7BB")


def text_slide(path: Path, title: str, lines: list[tuple[str, str, int, str]]) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    base(d)
    y = 230
    tfont = font("serif-bold", 76)
    for line in wrap(d, title, tfont, W - 144):
        d.text((72, y), line, font=tfont, fill=FG)
        y += 100
    y += 40
    for text, kind, size, color in lines:
        f = font(kind, size)
        for line in wrap(d, text, f, W - 144):
            d.text((72, y), line, font=f, fill=color)
            y += int(size * 1.45)
        y += 18
    img.save(path)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


FPS = 15


def segment(img: Path, wav: Path | None, dur: float, out: Path,
            bars: list[float] | None = None, pre: float = 0.0) -> None:
    vf = "format=yuv420p"
    if bars:  # frames with the played part of the waveform in the accent colour
        fdir = out.with_suffix("")
        fdir.mkdir(exist_ok=True)
        base_img = Image.open(img)
        a_dur = dur - pre - 0.6
        for k in range(int(dur * FPS) + 1):
            fr = base_img.copy()
            draw_bars(ImageDraw.Draw(fr), bars, min(1.0, max(0.0, (k / FPS - pre) / a_dur)))
            fr.save(fdir / f"{k:05d}.png", compress_level=1)
        cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(fdir / "%05d.png")]
    else:
        cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-t", f"{dur:.3f}", "-i", str(img)]
    if wav:
        cmd += ["-i", str(wav), "-af", f"adelay={int(pre * 1000)}:all=1,apad", "-t", f"{dur:.3f}"]
    else:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=24000:cl=mono"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", "30",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-shortest", str(out)]
    run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="teacher")
    ap.add_argument("--article", default="3")
    ap.add_argument("--countries", type=Path)
    ap.add_argument("--model-name", default="Africa TTS")
    ap.add_argument("--n-trained", type=int, default=92)
    ap.add_argument("--n-experimental", type=int, default=657)
    ap.add_argument("--langs", help="comma-separated codes, in this order (default: all)")
    a = ap.parse_args()
    meta = json.loads((a.samples / "meta.json").read_text())
    countries = json.loads(a.countries.read_text()) if a.countries else {}
    items = [it for it in meta["items"] if it["article"] == a.article and a.label in it["synth"]]
    if a.langs:
        order = a.langs.split(",")
        items = sorted((it for it in items if it["g2p"] in order), key=lambda it: order.index(it["g2p"]))
    tmp = Path(tempfile.mkdtemp(prefix="att_video_"))
    segs = []

    intro = tmp / "intro.png"
    text_slide(intro, f"One model. {a.n_trained} African languages.", [
        (f"{a.model_name} speaks {a.n_trained} African languages and English, all from one small "
         "open model.", "sans", 38, FG),
        (f"Listen to {len(items)} of them{'' if not a.langs else ' (more on the samples page)'} read the same line from the Universal Declaration of "
         "Human Rights:", "sans", 34, MUTED),
        (f"“{ART_EN[a.article]}”", "italic", 34, ACCENT)])
    segs.append(tmp / "000_intro.mp4")
    segment(intro, None, 5.0, segs[-1])

    for i, it in enumerate(items, 1):
        g = it["g2p"]
        wav = a.samples / "audio" / it["synth"][a.label]
        audio, sr = sf.read(wav, dtype="float32")
        name = NICE.get(it["language"], it["language"])
        country = COUNTRY_OVERRIDE.get(g) or " · ".join(CC.get(c, c) for c in countries.get(g, [])[:3])
        png = tmp / f"{i:03d}.png"
        bars = lang_slide(png, name, country, it["text"], ART_EN[a.article], audio, f"{i} / {len(items)}")
        dur = 0.4 + len(audio) / sr + 0.6
        segs.append(tmp / f"{i:03d}.mp4")
        segment(png, wav, dur, segs[-1], bars=bars, pre=0.4)
        print(f"{i}/{len(items)} {name}", flush=True)

    outro = tmp / "outro.png"
    text_slide(outro, f"…and {a.n_experimental} more languages to test", [
        ("The model reads IPA, so it can try any language africa-g2p can transcribe. "
         "Native speakers: try yours and tell us how it sounds.", "sans", 36, FG),
        ("Open source · free to use for non-commercial work · contributions welcome", "sans", 30, MUTED),
        ("github.com/AfriSpeech/africa-tts", "sans-bold", 40, ACCENT),
        ("huggingface.co/AfriSpeech/africa-tts", "sans-bold", 34, ACCENT)])
    segs.append(tmp / "999_outro.mp4")
    segment(outro, None, 7.0, segs[-1])

    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         "-movflags", "+faststart", str(a.out)])
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", str(a.out)], capture_output=True,
                                 text=True).stdout)
    print(f"wrote {a.out}: {len(items)} languages, {total / 60:.1f} min (frames in {tmp})")


if __name__ == "__main__":
    main()
