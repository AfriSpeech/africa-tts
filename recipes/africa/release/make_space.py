"""Build the static HF Space for the Africa TTS samples from the UDHR demo folder.

Usage: python make_space.py DEMO_DIR SPACE_DIR
  DEMO_DIR: meta.json + audio/*.wav (from synth_udhr.py). Audio is transcoded to MP3 (ffmpeg).
"""

import html
import json
import subprocess
import sys
from pathlib import Path

LABELS = {"teacher": ("Africa TTS", "AfriSpeech/africa-tts"),
          "turbo": ("Africa TTS Turbo", "AfriSpeech/africa-tts-turbo")}
NICE = {"english_eng": "English", "swahili_swa": "Swahili", "twi_twi": "Twi", "yoruba_yor": "Yoruba",
        "xhosa_xho": "Xhosa"}
ART_EN = {"1": "All human beings are born free and equal in dignity and rights. They are endowed "
               "with reason and conscience and should act towards one another in a spirit of brotherhood.",
          "3": "Everyone has the right to life, liberty and security of person.",
          "5": "No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment."}

CSS = """
:root{--bg:#fbfaf7;--fg:#1b1a17;--muted:#6b675f;--card:#fff;--line:#e7e3da;--accent:#a0471d;
--chip:#f4ece4;--sel:#1b1a17;--selfg:#fbfaf7}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141311;--fg:#ece8e1;--muted:#a19b90;
--card:#1c1b18;--line:#2f2c27;--accent:#e59866;--chip:#2a221c;--sel:#ece8e1;--selfg:#141311}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 16px 80px}
header h1{font:600 34px/1.15 Georgia,"Times New Roman",serif;margin:0 0 10px;letter-spacing:-.01em}
header p{color:var(--muted);max-width:720px;margin:0 0 6px}
header a{color:var(--accent)}
.models{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 0}
.models a{font-size:13px;padding:5px 12px;border-radius:999px;background:var(--chip);color:var(--accent);text-decoration:none}
.layout{display:grid;grid-template-columns:240px 1fr;gap:28px;margin-top:32px}
aside{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);display:flex;flex-direction:column}
aside input{width:100%;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);
color:var(--fg);font:inherit;margin-bottom:10px}
aside ul{list-style:none;margin:0;padding:0;overflow:auto;border-top:1px solid var(--line)}
aside li button{all:unset;cursor:pointer;display:flex;justify-content:space-between;width:100%;padding:7px 10px;
border-radius:6px;font-size:14px;box-sizing:border-box}
aside li button:hover{background:var(--chip)}
aside li button[aria-current=true]{background:var(--sel);color:var(--selfg)}
aside li .c{font:12px ui-monospace,monospace;opacity:.65}
main h2{font:600 26px/1.2 Georgia,serif;margin:0}
.meta{color:var(--muted);font-size:13px;margin:4px 0 18px}
.voice{border:1px dashed var(--line);border-radius:10px;padding:8px 14px;margin-top:18px;color:var(--muted);font-size:13px}
.voice summary{cursor:pointer}
.voice p{margin:8px 0}
.art{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:14px}
.art h3{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent)}
.art .t{font-size:17px;margin:0 0 12px}
.art .en{font-size:13px;color:var(--muted);margin:12px 0 4px;font-style:italic}
.ipa-d{font-size:12px;color:var(--muted)}.ipa-d summary{cursor:pointer}
.art .ipa{font:13.5px ui-monospace,"Noto Sans Mono",monospace;color:var(--muted);margin:6px 0 0;word-break:break-word}
.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.players div{border:1.5px solid var(--accent);border-radius:10px;padding:10px 12px;background:var(--chip)}
.players b{display:block;font-size:13px;margin-bottom:6px;color:var(--accent)}
audio{width:100%;height:40px}
.voice audio{height:34px}
footer{margin-top:48px;color:var(--muted);font-size:13px}
@media (max-width:760px){.layout{grid-template-columns:1fr}aside{position:static;max-height:none}
aside ul{max-height:220px}header h1{font-size:27px}}
"""

JS = """
const D = JSON.parse(document.getElementById('data').textContent);
const list = document.getElementById('list'), q = document.getElementById('q'), main = document.getElementById('main');
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function show(code, push) {
  const L = D.langs.find(l => l.code === code) || D.langs[0];
  list.querySelectorAll('button').forEach(b => b.setAttribute('aria-current', b.dataset.code === L.code));
  let h = `<h2>${esc(L.name)}</h2><div class="meta"><code>${L.code}</code> · ${L.hours.toFixed(1)} h of training speech</div>`;

  for (const a of L.articles) {
    h += `<section class="art"><h3>Article ${a.n}</h3><p class="t">${esc(a.text)}</p>`;
    h += `<div class="players">`;
    for (const [k, name] of D.labels) if (a.audio[k]) h += `<div><b>▶ ${esc(name)}</b><audio controls preload="metadata" src="audio/${a.audio[k]}"></audio></div>`;
    h += `</div>`;
    if (L.code !== 'eng') h += `<p class="en">${esc(D.en[a.n])}</p>`;
    h += `<details class="ipa-d"><summary>IPA read by the model</summary><p class="ipa">${esc(a.ipa)}</p></details></section>`;
  }
  h += `<details class="voice"><summary>Voice prompt used for cloning</summary><p>The first ≤ 5 s of a ${esc(L.name)} clip from the training data; every sample above is read in this voice.</p><audio controls preload="none" src="audio/${L.voice}"></audio></details>`;
  main.innerHTML = h;
  if (push) history.replaceState(null, '', '#' + L.code);
}
list.innerHTML = D.langs.map(l => `<li><button data-code="${l.code}"><span>${esc(l.name)}</span><span class="c">${l.code}</span></button></li>`).join('');
list.addEventListener('click', e => { const b = e.target.closest('button'); if (b) show(b.dataset.code, true); });
q.addEventListener('input', () => { const s = q.value.toLowerCase();
  list.querySelectorAll('li').forEach(li => li.hidden = !li.textContent.toLowerCase().includes(s)); });
show(location.hash.slice(1) || 'twi', false);
"""


def nice(n: str) -> str:
    if n in NICE:
        return NICE[n]
    return n.split("_")[0].capitalize() if "_" in n else n


def mp3(src: Path, dst: Path) -> str:
    if not dst.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame",
                        "-b:a", "96k", str(dst)], check=True)
    return dst.name


def main() -> None:
    demo, space = Path(sys.argv[1]), Path(sys.argv[2])
    (space / "audio").mkdir(parents=True, exist_ok=True)
    meta = json.loads((demo / "meta.json").read_text())
    hours = {}
    hp = demo / "lang_hours.json"
    if hp.exists():
        hours = {g: h for g, _, h, _ in json.loads(hp.read_text())["langs"]}
    langs: dict[str, dict] = {}
    for it in meta["items"]:
        g = it["g2p"]
        L = langs.setdefault(g, {"code": g, "name": nice(it["language"]), "hours": hours.get(g, 0.0),
                                 "voice": mp3(demo / "audio" / it["voice"],
                                              space / "audio" / (Path(it["voice"]).stem + ".mp3")),
                                 "articles": []})
        L["articles"].append({"n": it["article"], "text": it["text"], "ipa": it["ipa"],
                              "audio": {k: mp3(demo / "audio" / f, space / "audio" / (Path(f).stem + ".mp3"))
                                        for k, f in it["synth"].items()}})
    labels = [(k, LABELS[k][0]) for k in meta["labels"]]
    data = {"labels": labels, "en": ART_EN, "langs": list(langs.values())}
    n_african = sum(1 for g in langs if g != "eng")
    model_links = "".join(f"<a href='https://huggingface.co/{LABELS[k][1]}'>{html.escape(LABELS[k][0])} →</a>"
                          for k in meta["labels"])
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<" + chr(92) + "/")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Africa TTS samples</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>Africa TTS: one model, {n_african} African languages</h1>
<p>Articles 1, 3 and 5 of the Universal Declaration of Human Rights, synthesised by AfriSpeech's
Pocket TTS models in every training language with a complete UDHR translation. None of this text was in
the training data. Each language is read in a voice cloned from a short clip of that language;
the model reads IPA produced by <a href="https://github.com/AfriSpeech/africa-g2p">africa-g2p</a>.</p>
<div class="models">{model_links}</div></header>
<div class="layout"><aside><input id="q" type="search" placeholder="Search {len(langs)} languages" aria-label="Search languages">
<ul id="list"></ul></aside><main id="main"></main></div>
<footer>UDHR texts: <a href="https://github.com/eric-muller/udhr">Unicode UDHR collection</a>. Models: CC-BY-NC-4.0,
built on <a href="https://github.com/kyutai-labs/pocket-tts">Kyutai Pocket TTS</a>. Generation: temperature 0.3,
1 flow step; Africa TTS with guidance 2.0, Turbo with guidance baked in.</footer></div>
<script type="application/json" id="data">{data_json}</script>
<script>{JS}</script></body></html>"""
    (space / "index.html").write_text(page, encoding="utf-8")
    (space / "README.md").write_text(
        "---\ntitle: Africa TTS samples\nemoji: 🔊\ncolorFrom: yellow\ncolorTo: red\nsdk: static\n"
        "pinned: false\nlicense: cc-by-nc-4.0\nshort_description: UDHR read by Africa TTS in "
        f"{len(langs)} languages\n---\n\nListening samples for "
        + " and ".join(f"[{LABELS[k][0]}](https://huggingface.co/{LABELS[k][1]})" for k in meta["labels"])
        + ".\n", encoding="utf-8")
    print(f"wrote {space}/index.html: {len(langs)} languages, labels {meta['labels']}")


if __name__ == "__main__":
    main()
