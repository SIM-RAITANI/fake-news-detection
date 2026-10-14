"""Draw the README diagrams: system architecture + an example knowledge graph.

Run:  python src/make_diagrams.py
Output: results/figures/architecture.png, results/figures/knowledge_graph_example.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def box(ax, x, y, w, h, text, face, edge="#333333", fontsize=9, bold=False,
        dashed=False, text_color="#111111"):
    """Rounded box centred at (x, y)."""
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.3,rounding_size=0.8",
                           facecolor=face, edgecolor=edge, linewidth=1.6,
                           linestyle="--" if dashed else "-", zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4,
            color=text_color, fontweight="bold" if bold else "normal", wrap=True)
    return patch


def arrow(ax, start, end, color="#444444", label=None, dashed=False, rad=0.0,
          fontsize=7.5, lw=1.5, patchA=None, patchB=None):
    """Arrow from start to end; patchA/patchB clip it to the edges of node boxes."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=lw, zorder=2,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}",
                                 patchA=patchA, patchB=patchB, shrinkA=2, shrinkB=2))
    if label:
        # midpoint of the arc3 curve (exact for straight lines, close enough for curves)
        dx, dy = end[0] - start[0], end[1] - start[1]
        mx = (start[0] + end[0]) / 2 + 0.5 * rad * dy
        my = (start[1] + end[1]) / 2 - 0.5 * rad * dx
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize, zorder=5,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.9))


# --------------------------------------------------------------------------- #
# 1. Architecture diagram
# --------------------------------------------------------------------------- #
def draw_architecture():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # Phase backgrounds
    phases = [
        (1, 43, 98, 16, "#E8F1FB", "#4C72B0", "① OFFLINE — build the knowledge graph (runs once, training claims only)"),
        (1, 15, 98, 25, "#EAF7EE", "#2E8B57", "② ONLINE — predict a new claim (Graph RAG)"),
        (1, 1, 98, 12, "#FFF4E5", "#D9822B", "③ EVALUATION — same test claims for every system"),
    ]
    for x, y, w, h, face, edge, title in phases:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, linewidth=2, zorder=0))
        ax.text(x + 1, y + h - 1.3, title, fontsize=12.5, fontweight="bold", color=edge,
                va="center", zorder=1)

    # ① Offline row
    y1 = 49.5
    offline = [
        (11, "LIAR2 dataset\n(PolitiFact)\n\ntrain 18,369\nval 2,297 · test 2,296", "#C9DCF2"),
        (33, "Preprocessing\n\n• 6 labels → FAKE / REAL\n  (0–2 FAKE, 3–5 REAL)\n• clean text, drop \"Says\"\n• split subjects on \";\"\n• leakage fixes", "#C9DCF2"),
        (57, "NLP enrichment\n\n• spaCy NER → entities\n• justification → evidence\n  sentences (train only)\n• MiniLM embeddings\n  (384 numbers / claim)", "#C9DCF2"),
        (84, "Neo4j KNOWLEDGE GRAPH\n\nClaim · Speaker · Party · State\nSubject · Context · Entity · Evidence\n\nMERGE → one connected graph\n+ HNSW vector index", "#4C72B0"),
    ]
    for i, (x, text, face) in enumerate(offline):
        last = i == len(offline) - 1
        box(ax, x, y1, 18 if not last else 22, 11, text, face, edge="#4C72B0",
            fontsize=9.5, bold=last, text_color="white" if last else "#111111")
        if i:
            prev = offline[i - 1][0]
            arrow(ax, (prev + 9.6, y1), (x - (11.6 if last else 9.6), y1), color="#4C72B0", lw=2.2)

    # ② Online row
    y2 = 25.5
    online = [
        (8, "NEW CLAIM\n\n+ speaker\n+ context\n+ subjects", "white"),
        (21.5, "Embed + NER\n\nMiniLM vector\nspaCy entities", "#BFE6CC"),
        (35, "Vector search\n\ntop-50 similar\ntraining claims\n(HNSW index)", "#BFE6CC"),
        (50, "Graph expansion\n+ HYBRID SCORE\n\n0.6 × cosine sim\n+ 0.2 × same speaker/subject\n+ 0.2 × shared entities", "#BFE6CC"),
        (65.5, "Balanced pick\n\ntop 5 FAKE\n+ top 5 REAL\n+ their evidence", "#BFE6CC"),
        (79, "LLM prompt\n\nLlama 3.1 8B\n(Groq / Ollama)\ntemperature 0", "#BFE6CC"),
        (92, "OUTPUT\n\nFAKE / REAL\n+ 2–3 sentence\nexplanation", "#2E8B57"),
    ]
    widths = [9.5, 10, 10, 13, 10.5, 10, 9]
    for i, ((x, text, face), w) in enumerate(zip(online, widths)):
        last = i == len(online) - 1
        box(ax, x, y2, w, 12, text, face, edge="#2E8B57", fontsize=9.3,
            bold=(i in (0, len(online) - 1)), dashed=(i == 0),
            text_color="white" if last else "#111111")
        if i:
            px, pw = online[i - 1][0], widths[i - 1]
            arrow(ax, (px + pw / 2 + 0.6, y2), (x - w / 2 - 0.6, y2), color="#2E8B57", lw=2.2)

    # Graph feeds the online steps
    arrow(ax, (80, 43.6), (36, 32.2), color="#4C72B0", dashed=True, rad=0.15,
          label="vector index lookup", fontsize=9)
    arrow(ax, (86, 43.6), (54, 32.2), color="#4C72B0", dashed=True, rad=-0.1,
          label="walk edges: speaker, subjects,\nentities, evidence", fontsize=9)

    # ③ Evaluation row
    y3 = 6
    evals = [
        (14, "System 1: LLM alone\n(no examples)", "#FFE2BF"),
        (35, "System 2: Normal RAG\n(top-10 by text similarity)", "#FFE2BF"),
        (57, "System 3: Graph RAG (ours)\n(graph score, 5+5, evidence)", "#F5B971"),
        (82, "Metrics: Accuracy · Precision · Recall\nF1 (FAKE) · Macro-F1 · MCC\n+ McNemar significance test", "#D9822B"),
    ]
    for i, (x, text, face) in enumerate(evals):
        last = i == len(evals) - 1
        box(ax, x, y3, 18 if not last else 26, 5.5, text, face, edge="#D9822B",
            fontsize=9.5, bold=(i >= 2), text_color="white" if last else "#111111")
    arrow(ax, (66.4, y3), (68.6, y3), color="#D9822B", lw=2.2)
    arrow(ax, (92, 19.2), (88, 9.2), color="#D9822B", dashed=True,
          label="predictions on\ntest set", fontsize=9)

    ax.set_title("Fake News Detection using Graph RAG — System Architecture",
                 fontsize=18, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Example knowledge graph
# --------------------------------------------------------------------------- #
NODE_STYLE = {   # type: (face colour, text colour)
    "FAKE":     ("#E45756", "white"),
    "REAL":     ("#54A24B", "white"),
    "NEW":      ("white", "#111111"),
    "Speaker":  ("#4C78A8", "white"),
    "Party":    ("#9D6FD8", "white"),
    "State":    ("#F58518", "white"),
    "Subject":  ("#2CB5A8", "white"),
    "Context":  ("#8C8C8C", "white"),
    "Entity":   ("#EECA3B", "#111111"),
    "Evidence": ("#FF9DA6", "#111111"),
}

EDGE_COLOR = {
    "SAID": "#4C78A8", "MEMBER_OF": "#9D6FD8", "FROM": "#F58518", "ABOUT": "#2CB5A8",
    "SAID_IN": "#8C8C8C", "MENTIONS": "#B8960C", "HAS_EVIDENCE": "#E0607E",
}


def draw_knowledge_graph():
    # name: (x, y, type, text, width)
    nodes = {
        "dem":   (12, 76, "Party", "Party:\nDemocrat", 9),
        "rep":   (88, 76, "Party", "Party:\nRepublican", 10),
        "wi_st": (50, 79, "State", "State:\nWisconsin", 9),
        "tx_st": (78, 2, "State", "State:\nTexas", 8),
        "spX":   (24, 64, "Speaker", "Speaker:\nCandidate X", 11),
        "spY":   (76, 64, "Speaker", "Speaker:\nOfficial Y", 11),
        "spZ":   (57, 2, "Speaker", "Speaker:\nPundit Z", 10),
        "B":     (22, 46, "FAKE", "Claim B  [FAKE]\n\"Walker's tax cuts cost\nWisconsin 40,000 jobs.\"", 17),
        "A":     (78, 46, "REAL", "Claim A  [REAL]\n\"Wisconsin added jobs every\nmonth under Scott Walker.\"", 18),
        "C":     (34, 13, "FAKE", "Claim C  [FAKE]\n\"Texas lost 50,000 jobs\nafter its tax cuts.\"", 15),
        "D":     (80, 14, "REAL", "Claim D  [REAL]\n\"Unemployment in Ohio\nrose last year.\"", 15),
        "NEW":   (50, 62, "NEW", "NEW CLAIM  [ ? ]\n\"Scott Walker's tax cuts cost\nWisconsin 50,000 jobs.\"", 19),
        "taxes": (30, 28, "Subject", "Subject:\ntaxes", 8),
        "jobs":  (68, 28, "Subject", "Subject:\njobs", 8),
        "econ":  (93, 30, "Subject", "Subject:\neconomy", 9),
        "walker": (50, 48, "Entity", "Entity:\nScott Walker", 10),
        "wi_en": (50, 37, "Entity", "Entity:\nWisconsin", 9),
        "tx_en": (12, 13, "Entity", "Entity:\nTexas", 8),
        "oh_en": (96, 4, "Entity", "Entity:\nOhio", 7),
        "tvad":  (6, 33, "Context", "Context:\nTV ad", 8),
        "speech": (95, 50, "Context", "Context:\na speech", 9),
        "tweet": (57, 15, "Context", "Context:\na tweet", 8),
        "evB":   (6, 60, "Evidence", "Evidence:\n\"Job losses were\nunder 10,000.\"", 11),
        "evA":   (94, 63, "Evidence", "Evidence:\n\"BLS data show\nmonthly gains.\"", 11),
    }
    edges = [  # (src, dst, type, curvature)
        ("spX", "dem", "MEMBER_OF", 0), ("spY", "rep", "MEMBER_OF", 0),
        ("spX", "wi_st", "FROM", 0), ("spY", "wi_st", "FROM", 0), ("spZ", "tx_st", "FROM", 0),
        ("spX", "B", "SAID", 0), ("spY", "A", "SAID", 0),
        ("spZ", "C", "SAID", 0), ("spZ", "D", "SAID", 0),
        ("B", "taxes", "ABOUT", 0), ("C", "taxes", "ABOUT", 0), ("C", "jobs", "ABOUT", 0),
        ("A", "jobs", "ABOUT", 0), ("A", "econ", "ABOUT", 0), ("D", "econ", "ABOUT", 0),
        ("B", "walker", "MENTIONS", 0), ("B", "wi_en", "MENTIONS", 0),
        ("A", "walker", "MENTIONS", 0), ("A", "wi_en", "MENTIONS", 0),
        ("C", "tx_en", "MENTIONS", 0), ("D", "oh_en", "MENTIONS", 0),
        ("B", "tvad", "SAID_IN", 0), ("A", "speech", "SAID_IN", 0),
        ("C", "tweet", "SAID_IN", 0), ("D", "tweet", "SAID_IN", 0),
        ("B", "evB", "HAS_EVIDENCE", 0), ("A", "evA", "HAS_EVIDENCE", 0),
    ]
    query_edges = ["taxes", "jobs", "walker", "wi_en"]  # where the new claim plugs in

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 84)
    ax.axis("off")

    patches = {}
    for name, (x, y, typ, text, w) in nodes.items():
        face, tcol = NODE_STYLE[typ]
        h = 7.5 if typ in ("FAKE", "REAL", "NEW") else (6.5 if typ == "Evidence" else 4.6)
        patches[name] = box(ax, x, y, w, h, text, face,
                            edge="#111111" if typ == "NEW" else face, fontsize=9.5,
                            bold=typ in ("FAKE", "REAL", "NEW"), dashed=typ == "NEW",
                            text_color=tcol)

    def pos(n):
        return nodes[n][0], nodes[n][1]

    for src, dst, typ, rad in edges:
        arrow(ax, pos(src), pos(dst), color=EDGE_COLOR[typ], label=typ, rad=rad, lw=1.8,
              patchA=patches[src], patchB=patches[dst])
    for dst in query_edges:
        arrow(ax, pos("NEW"), pos(dst), color="#111111", dashed=True, lw=1.6,
              rad=0.45 if dst == "wi_en" else 0, patchA=patches["NEW"], patchB=patches[dst])

    legend = [Patch(facecolor=NODE_STYLE[t][0], edgecolor="#333333", label=lbl) for t, lbl in [
        ("FAKE", "Claim (FAKE)"), ("REAL", "Claim (REAL)"), ("Speaker", "Speaker"),
        ("Party", "Party"), ("State", "State"), ("Subject", "Subject"),
        ("Entity", "Entity (spaCy NER)"), ("Context", "Context"), ("Evidence", "Evidence sentence"),
    ]]
    legend.append(Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.6,
                         label="New claim → shared nodes\n(query time, not stored)"))
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=5,
              fontsize=10.5, frameon=True, title="Node types", title_fontsize=11)

    ax.set_title("Example Knowledge Graph (illustrative claims) — how one new claim connects to "
                 "past fact-checked claims", fontsize=15, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "knowledge_graph_example.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_architecture()
    draw_knowledge_graph()
    print(f"Saved diagrams to {FIG_DIR}")


if __name__ == "__main__":
    main()
