"""Generate the paper's LaTeX tables and headline numbers from the benchmark results.

Usage: python paper/make_tables.py RESULTS.json SPEED.json OUT_DIR
Writes OUT_DIR/{speed,averages,perlang}.tex and OUT_DIR/numbers.tex (\\newcommand macros).
Best synthetic system per row in bold (real speech is a reference, never bolded).
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OURS = ["turbo", "guided", "unguided"]
BASE = ["mms", "omnivoice", "orpheus"]
NAME = {"real": "Real", "turbo": "Turbo", "guided": "Guided", "unguided": "Unguided",
        "mms": "MMS-TTS", "omnivoice": "OmniVoice", "orpheus": "Orpheus"}
SIZE = {"real": "speech", "turbo": "110M", "guided": "336M", "unguided": "336M",
        "mms": "36M/lang.", "omnivoice": "813M", "orpheus": "3.3B"}
SYN = OURS + BASE
COLS = ["real"] + SYN  # narrowed to the systems present in the results by set_cols()


def set_cols(*results) -> None:
    global COLS, SYN
    present = set().union(*(r["systems"] for r in results))
    COLS = [s for s in ["real"] + OURS + BASE if s in present or s == "real"]
    SYN = [s for s in COLS if s != "real"]


def head(left: list[str], ours_label: str = "(ours)") -> str:
    """Two header rows: an 'Africa TTS' span over our three modes, then system names and sizes."""
    k = len(left)
    first = COLS.index("turbo") + k + 1
    pre = " & " * (k + COLS.index("turbo"))
    post = " & " * (len(COLS) - COLS.index("unguided") - 1)
    lines = [pre + r"\multicolumn{3}{c}{\textbf{Africa TTS} " + ours_label + "}" + post + r" \\",
             rf"\cmidrule(lr){{{first}-{first + 2}}}",
             " & ".join([rf"\textbf{{{x}}}" if x else "" for x in left]
                        + [rf"\textbf{{{NAME[c]}}}" for c in COLS]) + r" \\",
             " & ".join([""] * k + [rf"\textit{{{SIZE[c]}}}" for c in COLS]) + r" \\"]
    return "\n".join(lines)




def ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def cells(vals: dict, cols: list[str], blank: str = "--") -> list[str]:
    shown = {s: f"{100 * v:.1f}" for s, v in vals.items() if ok(v)}
    cands = [float(shown[s]) for s in cols if s in shown and s != "real"]
    best = min(cands) if len(cands) > 1 else None
    out = []
    for s in cols:
        if s not in shown:
            out.append(blank)
        elif s != "real" and best is not None and float(shown[s]) == best:
            out.append(r"\textbf{" + shown[s] + "}")
        else:
            out.append(shown[s])
    return out


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def src_short(s: str) -> str:
    return {"fleurs/test": "FL", "flores_plus/devtest": "FR"}.get(s, "") or (
        "OC" if s.startswith("omnilingual") else "WX" if s.startswith("waxal") else s)


def speed_table(sp: dict) -> str:
    r = sp["runs"]
    rows = [("Africa TTS Turbo", "sherpa-onnx int8", r.get("turbo_sherpa_int8")),
            ("Africa TTS Turbo", "PyTorch", r.get("turbo_python")),
            ("Africa TTS unguided", "sherpa-onnx int8", r.get("24l_unguided_sherpa_int8")),
            ("Africa TTS unguided", "PyTorch", r.get("24l_unguided_python")),
            ("Africa TTS guided", "PyTorch", r.get("24l_guided_python"))]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrr}", r"\toprule",
             r"\textbf{Model} & \textbf{Runtime} & \textbf{$\times$RT} $\uparrow$ & \textbf{TTFA} $\downarrow$ \\",
             r"\midrule"]
    best = max(v["x_realtime"] for _, _, v in rows if v)
    for m, rt, v in rows:
        if not v:
            continue
        x = f"{v['x_realtime']:.2f}"
        x = r"\textbf{" + x + "}" if v["x_realtime"] == best else x
        f = f"{v['first_audio_s']:.2f} s" if "first_audio_s" in v else "--"
        lines.append(f"{m} & {rt} & {x} & {f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{CPU speed on a dedicated 4-vCPU cloud machine: seconds of audio generated per"
              r" second of compute ($\times$RT, above 1 is faster than real time) and median time to"
              r" first audio when streaming (TTFA). PyTorch runs on one thread, sherpa-onnx on four.}",
              r"\label{tab:speed}", r"\end{table}"]
    return "\n".join(lines) + "\n"


LABEL = {"mms": "MMS-TTS", "omnivoice": "OmniVoice", "orpheus": "Sunbird Orpheus"}


def averages_table(res: dict) -> str:
    nc = len(COLS)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{l" + "r" * nc + "}",
             r"\toprule", head([""]), r"\midrule"]
    for key, title in (("cer", "CER (\\%)"), ("cer_nodiac", "CER without diacritics (\\%)")):
        lines.append(rf"\multicolumn{{{nc + 1}}}{{l}}{{\textit{{{title}}}}} \\")
        for comp in res["averages"]:
            n = len(comp["languages"])
            if n == 0:
                continue
            lab = (f"Languages {LABEL[comp['vs']]} supports" if comp["vs"] in LABEL else
                   "Languages with real recordings" if comp["vs"] == "real" else "All systems")
            lines.append(f"{lab} ({n}) & " + " & ".join(cells(comp[key], COLS, "")) + r" \\")
        if key == "cer":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Intelligibility on languages Africa TTS was trained on: mean per-language"
              r" character error rate of Omnilingual ASR transcripts (lower is better), averaged over"
              r" the languages each comparison shares (count in brackets; languages with fewer than"
              r" 10 test sentences are excluded). Best synthetic system in bold; real speech is the"
              r" ASR's error on human recordings of the same sentences. Sizes are parameter counts"
              r" including each system's audio codec or vocoder.}",
              r"\label{tab:averages}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def perlang_table(res: dict) -> str:
    lines = [r"\begin{table}[h!]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llcr" + "r" * len(COLS) + "}", r"\toprule",
             head(["Language", "Code", "Src", "$n$"]), r"\midrule"]
    for c, v in res["languages"].items():
        n = max(v[s]["n"] for s in SYN if s in v)
        note = r"$^\dagger$" if v.get("flags") else ""
        lines.append(f"{tex(v['name'])}{note} & \\texttt{{{c}}} & {src_short(v['source'])} & {n} & "
                     + " & ".join(cells({s: v[s]["cer"] for s in COLS if s in v}, COLS)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{CER (\%) per trained language. Src: OC Omnilingual ASR corpus, FL FLEURS,"
              r" WX WAXAL, FR FLORES+ (text only, no real recordings). $n$: sentences. $^\dagger$:"
              r" no Omnilingual ASR language code; transcribed without one. Best synthetic system in"
              r" bold.}",
              r"\label{tab:perlang}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def numbers(res: dict, sp: dict) -> str:
    import re
    L = res["languages"]
    macros = {"nTrainedLangs": len(L),
              "nSentences": sum(max(v[s]["n"] for s in SYN if s in v) for v in L.values())}

    def wins(a, b, langs):
        sh = [c for c in langs if a in langs[c] and b in langs[c]
              and max(langs[c][x]["n"] for x in SYN if x in langs[c]) >= 10]  # as in the means
        return sum(langs[c][a]["cer"] < langs[c][b]["cer"] for c in sh), len(sh)

    for a in OURS:
        for b in BASE:
            if b in SYN:
                w, n = wins(a, b, L)
                macros[f"{a}Beats{b}"] = f"{w} of {n}"
    for pre, r in (("avg", res),):
        for comp in r["averages"]:
            for s, v in comp["cer"].items():
                macros[f"{pre}{comp['vs']}{s}"] = f"{100 * v:.1f}"
            macros[f"{pre}N{comp['vs']}"] = len(comp["languages"])
    for k, v in sp["runs"].items():
        key = k.replace("24l", "full").replace("_", "")
        macros[f"xrt{key}"] = f"{v['x_realtime']:.2f}"
        if "first_audio_s" in v:
            macros[f"ttfa{key}"] = f"{v['first_audio_s']:.2f}"
    out = []
    for k, v in macros.items():
        name = re.sub(r"[^A-Za-z]", "", k.replace("omnivoice", "Omni").replace("mms", "Mms")
                      .replace("orpheus", "Orph"))
        out.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    return "\n".join(out) + "\n"


def main() -> None:
    res, sp = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    set_cols(res)
    out = Path(sys.argv[-1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "speed.tex").write_text(speed_table(sp))
    (out / "averages.tex").write_text(averages_table(res))
    (out / "perlang.tex").write_text(perlang_table(res))
    (out / "numbers.tex").write_text(numbers(res, sp))
    print((out / "numbers.tex").read_text())


if __name__ == "__main__":
    main()
