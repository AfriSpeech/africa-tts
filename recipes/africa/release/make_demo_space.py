"""Build the static HF Space for the live Africa TTS demo (AfriSpeech/africa-tts-demo).

The page calls the public API server (server/modal_app.py) from the browser: no secret is in the
page; the API protects itself with per-IP rate limits, a daily budget and a container cap.

Usage: python make_demo_space.py EXAMPLES.json SPACE_DIR --api https://...modal.run
  EXAMPLES.json: {lang: [{"text", "src"}, ...]} example sentences per trained language.
"""

import argparse
import json
from pathlib import Path

PKG = Path(__file__).resolve().parents[3] / "africa_tts"
VOICES = "https://huggingface.co/AfriSpeech/africa-tts/resolve/main/voices"

README = """---
title: Africa TTS live demo
emoji: 🗣️
colorFrom: yellow
colorTo: red
sdk: static
pinned: true
license: cc-by-nc-4.0
short_description: Type text in 92 African languages or English, hear it spoken
models:
- AfriSpeech/africa-tts
---

Live demo of [Africa TTS](https://huggingface.co/AfriSpeech/africa-tts). The page calls
a rate-limited public API ([server code](https://github.com/AfriSpeech/africa-tts/tree/main/server));
pre-generated samples in 53 languages are in
[africa-tts-samples](https://huggingface.co/spaces/AfriSpeech/africa-tts-samples).
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Africa TTS live demo</title>
<style>
:root{--bg:#fbfaf7;--fg:#1b1a17;--muted:#6b675f;--card:#fff;--line:#e7e3da;--accent:#a0471d;
--chip:#f4ece4;--warn:#8a5a00;--warnbg:#fff4dc;--err:#a3271d;--errbg:#fde9e6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141311;--fg:#ece8e1;--muted:#a19b90;
--card:#1c1b18;--line:#2f2c27;--accent:#e59866;--chip:#2a221c;--warn:#f0c46a;--warnbg:#2b2416;--err:#f19a8f;--errbg:#2e1a17}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:36px 16px 72px}
h1{font:600 34px/1.15 Georgia,"Times New Roman",serif;margin:0 0 8px;letter-spacing:-.01em}
.lede{color:var(--muted);margin:0 0 14px;max-width:680px}
a{color:var(--accent)}
.links{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 26px}
.links a{font-size:13px;padding:5px 12px;border-radius:999px;background:var(--chip);text-decoration:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
label.h{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--accent);margin:0 0 6px;font-weight:600}
.row{margin-bottom:18px}
select,textarea,input[type=search]{width:100%;font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:10px 12px}
textarea{min-height:110px;resize:vertical;font-size:17px;line-height:1.5}
.langbar{display:grid;grid-template-columns:1fr 1.4fr;gap:10px}
.count{font-size:12px;color:var(--muted);text-align:right;margin-top:4px}
.count.over{color:var(--err)}
.ex{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.ex button{all:unset;cursor:pointer;font-size:13px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);color:var(--muted);max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ex button:hover{border-color:var(--accent);color:var(--accent)}
.note{font-size:13px;border-radius:9px;padding:8px 12px;margin-top:8px}
.note.warn{background:var(--warnbg);color:var(--warn)}
.voice{display:flex;gap:18px;flex-wrap:wrap;align-items:center;font-size:14px}
.voice label{display:flex;gap:6px;align-items:center;cursor:pointer}
#own{margin-top:10px;display:none;gap:10px;flex-wrap:wrap;align-items:center;font-size:14px}
#own.on{display:flex}
.btn2{all:unset;cursor:pointer;font-size:14px;padding:6px 12px;border-radius:8px;border:1px solid var(--line)}
.btn2:hover{border-color:var(--accent)}
.btn2.rec{border-color:var(--err);color:var(--err)}
.small{font-size:12px;color:var(--muted)}
details.dv{margin-top:8px;font-size:13px;color:var(--muted)}
details.dv summary{cursor:pointer}
audio{width:100%;height:40px}
#go{all:unset;box-sizing:border-box;cursor:pointer;display:block;text-align:center;width:100%;padding:13px;border-radius:10px;background:var(--accent);color:#fff;font-weight:600;font-size:16px}
#go[disabled]{opacity:.55;cursor:wait}
#out{margin-top:18px;display:none}
#out.on{display:block}
.result{border:1.5px solid var(--accent);background:var(--chip);border-radius:12px;padding:14px}
.result .meta{display:flex;justify-content:space-between;gap:10px;font-size:13px;color:var(--muted);margin-top:6px;flex-wrap:wrap}
#status{font-size:14px;color:var(--muted);margin-top:12px;min-height:1.4em}
#status.err{color:var(--err)}
footer{margin-top:34px;color:var(--muted);font-size:13px}
footer ul{padding-left:18px;margin:6px 0}
@media (max-width:600px){h1{font-size:27px}.langbar{grid-template-columns:1fr}}
</style></head>
<body><div class="wrap">
<h1>Africa TTS, live</h1>
<p class="lede">Type a sentence in one of <b>92 African languages or English</b> and hear it spoken by
<b>Africa TTS</b>, a fast open text-to-speech model that reads African languages through
their pronunciation (IPA). Use each language's default voice, or clone your own from a short clip.</p>
<div class="links">
<a href="https://huggingface.co/AfriSpeech/africa-tts" target="_blank" rel="noopener">Model: africa-tts</a>
<a href="https://huggingface.co/spaces/AfriSpeech/africa-tts-samples" target="_blank" rel="noopener">Samples in 53 languages</a>
<a href="https://github.com/AfriSpeech/africa-tts" target="_blank" rel="noopener">Code &amp; API on GitHub</a>
</div>

<div class="card">
<div class="row">
<label class="h" for="lang">Language</label>
<div class="langbar">
<input type="search" id="find" placeholder="Search languages…" aria-label="Search languages">
<select id="lang"></select>
</div>
<div id="expnote" class="note warn" hidden></div>
</div>

<div class="row">
<label class="h" for="text">Text</label>
<textarea id="text" maxlength="__MAX__" spellcheck="false"></textarea>
<div class="count" id="count"></div>
<div class="small">Examples (click to use):</div>
<div class="ex" id="ex"></div>
</div>

<div class="row">
<label class="h">Voice</label>
<div class="voice">
<label><input type="radio" name="voice" value="default" checked> Default voice for this language</label>
<label><input type="radio" name="voice" value="own"> My own voice</label>
</div>
<details class="dv" id="dv"><summary>Hear the default voice</summary><audio id="dvaudio" controls preload="none"></audio></details>
<div id="own">
<button class="btn2" id="recbtn" type="button">● Record (up to 10 s)</button>
<span class="small">or</span>
<label class="btn2">Upload a clip<input type="file" id="file" accept="audio/*" hidden></label>
<span class="small" id="ownstate">3-10 s of clear speech, one speaker.</span>
<audio id="ownaudio" controls hidden></audio>
<span class="small">Your clip is used only to make this audio and is not saved.</span>
</div>
</div>

<button id="go" type="button">Speak</button>
<div id="status" role="status" aria-live="polite"></div>
<div id="out"><div class="result"><audio id="result" controls></audio>
<div class="meta"><span id="rmeta"></span><a id="dl" download="africa-tts.wav">Download WAV</a></div></div></div>
</div>

<footer>
<ul>
<li><b>Spell out numbers</b> ("twenty", not "20"): the model was trained without digits.</li>
<li>This is a free shared demo with limits (a few requests a minute, up to __MAX__ characters).
The server sleeps when idle, so the first request after a quiet spell takes ~15 s.
For heavier use, run the <a href="https://github.com/AfriSpeech/africa-tts/tree/main/server" target="_blank" rel="noopener">API server</a> yourself or use the
<a href="https://github.com/AfriSpeech/africa-tts" target="_blank" rel="noopener">Python package</a>.</li>
<li><b>Experimental</b> languages were not in the training data: they use africa-g2p's IPA and the voice of the closest trained language. Feedback welcome on <a href="https://github.com/AfriSpeech/africa-tts/issues" target="_blank" rel="noopener">GitHub</a>.</li>
<li>Models and voices: CC-BY-NC-4.0 (non-commercial). Built by AfriSpeech on Kyutai's Pocket TTS.</li>
</ul>
</footer>
</div>
<script>
const API = "__API__", MAX = __MAX__, VOICES = "__VOICES__";
const LANGS = __LANGS__, EXP = __EXP__, EX = __EXAMPLES__;
const $ = id => document.getElementById(id);
const sel = $("lang"), text = $("text");
let ownWav = null, warmed = false, lastUrl = null;

function buildOptions(q) {
  q = (q || "").toLowerCase().trim();
  const cur = sel.value;
  const match = (c, n) => !q || n.toLowerCase().includes(q) || c.includes(q);
  sel.innerHTML = "";
  const g1 = document.createElement("optgroup"); g1.label = "Trained";
  Object.entries(LANGS).filter(([c, n]) => match(c, n)).sort((a, b) => a[1].localeCompare(b[1]))
    .forEach(([c, n]) => g1.append(new Option(`${n} (${c})`, c)));
  const g2 = document.createElement("optgroup"); g2.label = "Experimental (not trained)";
  Object.entries(EXP).filter(([c, v]) => match(c, v[0])).sort((a, b) => a[1][0].localeCompare(b[1][0]))
    .forEach(([c, v]) => g2.append(new Option(`${v[0]} (${c})`, c)));
  if (g1.children.length) sel.append(g1);
  if (g2.children.length) sel.append(g2);
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
}

function langChanged(fill) {
  const c = sel.value;
  if (!c) return;
  const exp = EXP[c], voiceLang = exp ? exp[1] : c;
  $("expnote").hidden = !exp;
  if (exp) $("expnote").textContent = `Experimental: ${exp[0]} was not in the training data. ` +
    `It is read with africa-g2p's IPA and the voice of ${LANGS[exp[1]] || exp[1]}, the closest trained language.`;
  $("dvaudio").src = `${VOICES}/${voiceLang}.wav`;
  const exs = EX[c] || [];
  $("ex").innerHTML = "";
  exs.forEach(e => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = e.text; b.title = `${e.text}\n(${e.src})`;
    b.onclick = () => { text.value = e.text.slice(0, MAX); counter(); };
    $("ex").append(b);
  });
  if (!exs.length) $("ex").innerHTML = '<span class="small">No examples for this language: type your own sentence.</span>';
  if (fill && exs.length) { text.value = exs[0].text.slice(0, MAX); counter(); }
  try { localStorage.setItem("att-lang", c); } catch (e) {}
}

function counter() {
  const n = text.value.length;
  $("count").textContent = `${n} / ${MAX}`;
  $("count").classList.toggle("over", n > MAX);
}

function warm() {  // wake the API on first interaction (not on page load), so idle visits cost nothing
  if (warmed) return; warmed = true;
  fetch(`${API}/health`).catch(() => { warmed = false; });
}

// ---- own voice: record or upload, converted to 24 kHz mono WAV (<= 10 s) in the browser
function encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return new Uint8Array(buf);
}
async function toWav(blob) {
  const ctx = new AudioContext();
  const dec = await ctx.decodeAudioData(await blob.arrayBuffer());
  ctx.close();
  const sr = 24000, secs = Math.min(dec.duration, 10);
  const off = new OfflineAudioContext(1, Math.ceil(secs * sr), sr);
  const src = off.createBufferSource(); src.buffer = dec; src.connect(off.destination); src.start();
  const out = (await off.startRendering()).getChannelData(0);
  return { wav: encodeWav(out, sr), secs };
}
async function setOwn(blob) {
  try {
    const { wav, secs } = await toWav(blob);
    if (secs < 2) { $("ownstate").textContent = "Too short: use 3-10 s of speech."; ownWav = null; return; }
    ownWav = wav;
    const a = $("ownaudio"); a.src = URL.createObjectURL(new Blob([wav], { type: "audio/wav" })); a.hidden = false;
    $("ownstate").textContent = `Clip ready (${secs.toFixed(1)} s).`;
  } catch (e) { $("ownstate").textContent = "Could not read that audio file."; ownWav = null; }
}
function b64(bytes) {
  let s = ""; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
let rec = null;
$("recbtn").onclick = async () => {
  if (rec) { rec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = []; rec = new MediaRecorder(stream);
    rec.ondataavailable = e => chunks.push(e.data);
    rec.onstop = () => {
      stream.getTracks().forEach(t => t.stop()); clearTimeout(rec._t); rec = null;
      $("recbtn").textContent = "● Record again"; $("recbtn").classList.remove("rec");
      setOwn(new Blob(chunks, { type: chunks[0]?.type || "audio/webm" }));
    };
    rec.start(); rec._t = setTimeout(() => rec && rec.stop(), 10000);
    $("recbtn").textContent = "■ Stop"; $("recbtn").classList.add("rec");
    $("ownstate").textContent = "Recording… speak naturally.";
  } catch (e) { $("ownstate").textContent = "Microphone not available: upload a clip instead."; }
};
$("file").onchange = e => e.target.files[0] && setOwn(e.target.files[0]);
document.querySelectorAll("input[name=voice]").forEach(r => r.onchange = () => {
  const own = document.querySelector("input[name=voice]:checked").value === "own";
  $("own").classList.toggle("on", own); $("dv").hidden = own;
});

// ---- synthesis
async function speak() {
  const t = text.value.trim();
  const st = $("status"); st.className = "";
  if (!t) { st.textContent = "Type something first."; return; }
  if (t.length > MAX) { st.className = "err"; st.textContent = `Please keep it under ${MAX} characters.`; return; }
  const own = document.querySelector("input[name=voice]:checked").value === "own";
  if (own && !ownWav) { st.className = "err"; st.textContent = "Record or upload a clip for your voice first."; return; }
  const body = { text: t, lang: sel.value, format: "wav" };
  if (own) body.voice_wav_base64 = b64(ownWav);
  $("go").disabled = true; st.textContent = "Generating…";
  const slow = setTimeout(() => { st.textContent = "Waking the server up (first request after a quiet spell takes ~15 s)…"; }, 4000);
  try {
    const r = await fetch(`${API}/v1/tts`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) {
      let msg = `Error ${r.status}`;
      try { msg = (await r.json()).detail || msg; } catch (e) {}
      if (r.status === 429) { const s = +r.headers.get("Retry-After") || 60;
        msg = `${msg[0].toUpperCase() + msg.slice(1)}. Try again in ${s < 120 ? s + " s" : Math.ceil(s / 60) + " min"}.`; }
      else if (r.status === 503) msg = "The demo is busy right now. Please try again in a moment.";
      throw new Error(msg);
    }
    const blob = await r.blob();
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    lastUrl = URL.createObjectURL(blob);
    $("result").src = lastUrl; $("dl").href = lastUrl;
    $("dl").download = `africa-tts-${sel.value}.wav`;
    const a = r.headers.get("X-Audio-Seconds"), c = r.headers.get("X-Compute-Seconds");
    $("rmeta").textContent = a && c ? `${(+a).toFixed(1)} s of audio, generated in ${(+c).toFixed(1)} s on CPU` : "";
    $("out").classList.add("on"); st.textContent = "";
    $("result").play().catch(() => {});
  } catch (e) {
    st.className = "err";
    st.textContent = e.message === "Failed to fetch" ? "Could not reach the server. Check your connection and try again." : e.message;
  } finally { clearTimeout(slow); $("go").disabled = false; }
}

$("go").onclick = speak;
text.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) speak(); });
text.addEventListener("input", counter);
["focusin", "pointerdown"].forEach(ev => document.addEventListener(ev, warm, { once: true }));
$("find").addEventListener("input", e => {
  const before = sel.value; buildOptions(e.target.value); langChanged(sel.value !== before);
});
sel.addEventListener("change", () => langChanged(true));

buildOptions("");
let start = "twi"; try { start = localStorage.getItem("att-lang") || "twi"; } catch (e) {}
sel.value = (start in LANGS || start in EXP) ? start : "twi";
langChanged(true);
</script>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--api", required=True)
    ap.add_argument("--max-chars", type=int, default=500)
    a = ap.parse_args()
    langs = {c: v["name"] for c, v in json.loads((PKG / "languages.json").read_text()).items()}
    exp = {c: [v["name"], v["nearest"]]
           for c, v in json.loads((PKG / "languages_experimental.json").read_text()).items()}
    examples = json.loads(a.examples.read_text())
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")  # noqa: E731
    page = (PAGE.replace("__API__", a.api.rstrip("/")).replace("__MAX__", str(a.max_chars))
            .replace("__VOICES__", VOICES).replace("__LANGS__", dump(langs))
            .replace("__EXP__", dump(exp)).replace("__EXAMPLES__", dump(examples)))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "index.html").write_text(page)
    (a.out / "README.md").write_text(README)
    print(f"{a.out}/index.html: {len(langs)} trained + {len(exp)} experimental languages, "
          f"{sum(map(len, examples.values()))} examples")


if __name__ == "__main__":
    main()
