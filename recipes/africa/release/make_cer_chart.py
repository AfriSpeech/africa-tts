"""Per-language CER chart for the model card: one row per tested language (>= 10 sentences),
a bar per system that supports it (benchmarks/results/intelligibility.json).

    python recipes/africa/release/make_cer_chart.py benchmarks/results/intelligibility.json OUT.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SYSTEMS = [("turbo", "Africa TTS (110M)", "#2a78d6"),  # categorical slots 1-3, validated
           ("mms", "MMS-TTS (36M per language)", "#eb6834"),
           ("omnivoice", "OmniVoice (813M)", "#1baf7a")]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1e1c", "#6b675f", "#e7e3da"


def main() -> None:
    res = json.load(open(sys.argv[1]))
    langs = [(c, v) for c, v in res["languages"].items() if "turbo" in v
             and max(v[s]["n"] for s in v if isinstance(v[s], dict) and "n" in v[s] and s != "real") >= 10]
    langs.sort(key=lambda cv: cv[1]["turbo"]["cer"])
    row_h, bar_h = 1.0, 0.27
    fig, ax = plt.subplots(figsize=(8.5, 1.4 + 0.36 * len(langs)), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (c, v) in enumerate(langs):
        y = -i * row_h
        present = [s for s in SYSTEMS if s[0] in v]
        for k, (key, _, color) in enumerate(present):
            yy = y - (k - (len(present) - 1) / 2) * (bar_h + 0.03)  # Africa TTS on top; 0.03 gap
            cer = 100 * v[key]["cer"]
            ax.barh(yy, cer, height=bar_h, color=color, linewidth=0)
            if key == "turbo":
                ax.text(cer + 0.4, yy, f"{cer:.1f}", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks([-i * row_h for i in range(len(langs))])
    ax.set_yticklabels([v["name"] for _, v in langs], fontsize=8, color=INK)
    ax.set_ylim(-(len(langs) - 1) * row_h - 0.7, 0.7)
    ax.set_xlabel("character error rate of Omnilingual ASR on the speech (%, lower is better)",
                  fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SYSTEMS]
    ax.legend(handles, [n for _, n, _ in SYSTEMS], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, frameon=False, fontsize=8.5, handlelength=1.1, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(sys.argv[2], facecolor=SURFACE)
    print("wrote", sys.argv[2], len(langs), "languages")


if __name__ == "__main__":
    main()
