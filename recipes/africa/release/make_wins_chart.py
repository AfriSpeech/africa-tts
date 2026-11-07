"""Head-to-head bar chart for the model card: in how many languages each system has the lower
ASR character error rate (benchmarks/results/intelligibility.json, languages with >= 10 sentences).

    python recipes/africa/release/make_wins_chart.py benchmarks/results/intelligibility.json OUT.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OURS, OTHER = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated: CVD dE 24.7)
SURFACE, INK, MUTED = "#fcfcfb", "#1f1e1c", "#6b675f"
NAMES = {"mms": "MMS-TTS", "omnivoice": "OmniVoice"}
SIZES = {"mms": "36M per language", "omnivoice": "813M"}


def wins(res: dict, other: str) -> tuple[int, int]:
    langs = [v for v in res["languages"].values()
             if "turbo" in v and other in v
             and max(v[s]["n"] for s in v if isinstance(v[s], dict) and "n" in v[s] and s != "real") >= 10]
    w = sum(round(v["turbo"]["cer"], 4) < round(v[other]["cer"], 4) for v in langs)
    lose = sum(round(v["turbo"]["cer"], 4) > round(v[other]["cer"], 4) for v in langs)
    return w, lose


def seg(ax, x0, width, y, h, color):
    box = FancyBboxPatch((x0, y - h / 2), width, h, boxstyle="round,pad=0,rounding_size=0.25",
                         linewidth=0, facecolor=color, mutation_aspect=0.06)
    ax.add_patch(box)


def main() -> None:
    res = json.load(open(sys.argv[1]))
    rows = [(o, *wins(res, o)) for o in ("mms", "omnivoice") if o in res["systems"]]
    top = max(w + lose for _, w, lose in rows)
    xmax = 5 * (top // 5 + 1)
    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.8 * len(rows)), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    gap = 0.12  # surface gap between stacked segments, in languages
    for i, (o, w, lose) in enumerate(rows):
        y = len(rows) - 1 - i
        seg(ax, 0, w - gap / 2, y, 0.5, OURS)            # Africa TTS wins first
        seg(ax, w + gap / 2, lose - gap / 2, y, 0.5, OTHER)  # then the other model's
        for x, val in ((w / 2, w), (w + lose / 2, lose)):
            if val >= 2:
                ax.text(x, y, str(val), ha="center", va="center", color="white",
                        fontsize=11.5, fontweight="bold")
        ax.text(w + lose + 0.4, y, f"{w + lose} languages", ha="left", va="center",
                color=MUTED, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"vs {NAMES[o]} ({SIZES[o]})" for o, _, _ in reversed(rows)],
                       fontsize=10.5, color=INK)
    ax.set_xlim(0, xmax + 4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(0, xmax + 1, 5))
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("languages", color=MUTED, fontsize=9)
    ax.grid(axis="x", color="#e7e3da", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [FancyBboxPatch((0, 0), 1, 1, facecolor=OURS, linewidth=0),
               FancyBboxPatch((0, 0), 1, 1, facecolor=OTHER, linewidth=0)]
    ax.legend(handles, ["Africa TTS (110M) better", "Other model better"], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5, handlelength=1.2,
              labelcolor=INK)
    fig.text(0.01, 0.02, "Languages where each system's speech gets the lower character error rate "
             "from Omnilingual ASR (held-out sentences, >= 10 per language).",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(sys.argv[2], facecolor=SURFACE)
    print("wrote", sys.argv[2], rows)


if __name__ == "__main__":
    main()
