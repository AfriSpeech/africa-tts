"""Model-card chart: CER per language, Africa TTS vs MMS-TTS, on the languages both support
(>= 10 sentences), lowest Africa TTS CER first (benchmarks/results/intelligibility.json).

    python recipes/africa/release/make_cer_chart.py benchmarks/results/intelligibility.json OUT.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS = ("turbo", "Africa TTS (110M)", "#2a78d6")  # categorical slots 1-2, validated
OTHER = ("mms", "MMS-TTS (36M per language)", "#eb6834")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1e1c", "#6b675f", "#e7e3da"


def main() -> None:
    res = json.load(open(sys.argv[1]))
    langs = [(c, v) for c, v in res["languages"].items() if OURS[0] in v and OTHER[0] in v
             and max(v[s]["n"] for s in v if isinstance(v[s], dict) and "n" in v[s] and s != "real") >= 10]
    langs.sort(key=lambda cv: cv[1][OURS[0]]["cer"])
    wins = sum(v[OURS[0]]["cer"] < v[OTHER[0]]["cer"] for _, v in langs)
    w, gap = 0.38, 0.03
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (_, v) in enumerate(langs):
        for k, (key, _, color) in enumerate((OURS, OTHER)):
            x = i + (k - 0.5) * (w + gap)
            ax.bar(x, 100 * v[key]["cer"], width=w, color=color, linewidth=0)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([v["name"].split(" (")[0] for _, v in langs], rotation=55, ha="right", fontsize=8.5,
                       color=INK)
    ax.set_xlim(-0.7, len(langs) - 0.3)
    ax.set_ylabel("character error rate (%)\n↓ lower is better", fontsize=9, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in (OURS, OTHER)]
    ax.legend(handles, [OURS[1], OTHER[1]], loc="upper left", frameon=False, fontsize=9.5,
              handlelength=1.1, labelcolor=INK)
    ax.set_title(f"Lower is better: Africa TTS has the lower error rate in {wins} of "
                 f"{len(langs)} languages", loc="left", fontsize=11, color=INK, pad=10)
    fig.text(0.01, 0.01, "Character error rate of Omnilingual ASR transcripts of each system's "
             "speech, held-out sentences (>= 10 per language); sorted by Africa TTS, lowest first.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(sys.argv[2], facecolor=SURFACE)
    print("wrote", sys.argv[2], len(langs), "languages,", wins, "wins")


if __name__ == "__main__":
    main()
