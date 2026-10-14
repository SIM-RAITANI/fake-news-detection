"""Step 1b: explore the raw LIAR2 data — counts, missing values, charts.

Run:  python src/explore_data.py
Output: printed statistics + charts in results/figures/
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "results" / "figures"

LABEL_NAMES = {0: "pants-fire", 1: "false", 2: "barely-true",
               3: "half-true", 4: "mostly-true", 5: "true"}


def to_binary(label):
    return "FAKE" if label <= 2 else "REAL"


def save_bar(series, title, xlabel, filename, horizontal=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    (series.plot.barh if horizontal else series.plot.bar)(ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel if not horizontal else "count")
    if horizontal:
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    splits = {s: pd.read_csv(RAW_DIR / f"{s}.csv") for s in ("train", "validation", "test")}
    train = splits["train"]

    print("=" * 60, "\n1. SIZE OF EACH SPLIT")
    for name, df in splits.items():
        print(f"  {name:<10} {len(df):>6} rows")

    print("=" * 60, "\n2. COLUMNS AND MISSING VALUES (train)")
    missing = train.isna().sum()
    empty = (train.astype(str).apply(lambda c: c.str.strip()) == "").sum()
    for col in train.columns:
        print(f"  {col:<22} dtype={str(train[col].dtype):<8} missing={missing[col]:>5}  empty={empty[col]:>5}")

    print("=" * 60, "\n3. LABEL DISTRIBUTION (6-way)")
    for name, df in splits.items():
        counts = df["label"].value_counts().sort_index()
        pretty = ", ".join(f"{LABEL_NAMES[k]}={v}" for k, v in counts.items())
        print(f"  {name:<10} {pretty}")

    print("=" * 60, "\n4. BINARY DISTRIBUTION (FAKE = labels 0-2, REAL = 3-5)")
    for name, df in splits.items():
        b = df["label"].apply(to_binary).value_counts()
        print(f"  {name:<10} FAKE={b.get('FAKE', 0)} ({b.get('FAKE', 0) / len(df):.1%})  "
              f"REAL={b.get('REAL', 0)} ({b.get('REAL', 0) / len(df):.1%})")

    print("=" * 60, "\n5. LABEL CHECK: one example per label (statement | end of justification)")
    for lab in range(6):
        row = train[train["label"] == lab].iloc[0]
        just = str(row["justification"])[-160:].replace("\n", " ")
        print(f"  [{lab} {LABEL_NAMES[lab]}] {str(row['statement'])[:90]}")
        print(f"      ...{just}")

    print("=" * 60, "\n6. TEXT LENGTHS (words, train)")
    for col in ("statement", "justification"):
        words = train[col].fillna("").astype(str).str.split().str.len()
        print(f"  {col:<14} mean={words.mean():.1f}  median={words.median():.0f}  max={words.max()}")

    print("=" * 60, "\n7. SPEAKERS, SUBJECTS, CONTEXTS (train)")
    speakers = train["speaker"].fillna("").str.lower().str.strip()
    subjects = (train["subject"].fillna("").str.lower().str.split(";").explode().str.strip())
    subjects = subjects[subjects != ""]
    contexts = train["context"].fillna("").str.lower().str.strip()
    states = train["state_info"].fillna("").str.lower().str.strip()
    print(f"  unique speakers : {speakers.nunique()}")
    print(f"  unique subjects : {subjects.nunique()}")
    print(f"  unique contexts : {contexts.nunique()}")
    print(f"  unique states   : {states.nunique()}")
    print(f"  subjects per claim: mean={train['subject'].fillna('').str.split(';').str.len().mean():.2f}")
    print("  top 10 speakers:", ", ".join(f"{k} ({v})" for k, v in speakers.value_counts().head(10).items()))
    print("  top 10 subjects:", ", ".join(f"{k} ({v})" for k, v in subjects.value_counts().head(10).items()))
    print("  top 10 contexts:", ", ".join(f"{k} ({v})" for k, v in contexts.value_counts().head(10).items()))
    once = (speakers.value_counts() == 1).sum()
    print(f"  speakers with only ONE claim: {once} ({once / speakers.nunique():.1%} of speakers)")

    print("=" * 60, "\n8. DATES")
    dates = pd.to_datetime(train["date"], errors="coerce", format="mixed")
    print(f"  parsed {dates.notna().sum()} / {len(train)} dates;  range {dates.min().date()} -> {dates.max().date()}")
    after = (dates >= "2024-01-01").sum()
    print(f"  claims from 2024 onwards (after Llama 3.1 knowledge cutoff): {after}")

    print("=" * 60, "\n9. DUPLICATES")
    norm = lambda s: s.fillna("").str.lower().str.strip()
    print(f"  exact duplicate statements inside train: {norm(train['statement']).duplicated().sum()}")
    train_set = set(norm(train["statement"]))
    for name in ("validation", "test"):
        overlap = norm(splits[name]["statement"]).isin(train_set).sum()
        print(f"  {name} statements that also appear in train: {overlap}")

    print("=" * 60, "\n10. SHORTCUT RISK: % FAKE for the biggest speakers / subjects (train)")
    is_fake = train["label"] <= 2
    print(f"  overall FAKE rate: {is_fake.mean():.1%}")
    for sp in speakers.value_counts().head(6).index:
        m = speakers == sp
        print(f"  speaker '{sp}': {m.sum()} claims, {is_fake[m].mean():.1%} FAKE")
    subj_lists = train["subject"].fillna("").str.lower().str.split(";")
    for subj in ("facebook fact-checks", "punditfact", "health care", "economy"):
        m = subj_lists.apply(lambda xs: subj in [x.strip() for x in xs])
        print(f"  subject '{subj}': {m.sum()} claims, {is_fake[m].mean():.1%} FAKE")
    says = train["statement"].str.match(r"^\s*(Says|Said)\b")
    print(f"  statements starting with 'Says/Said': {says.sum()} ({says.mean():.1%})")

    # ---------------- charts ----------------
    lab_counts = train["label"].value_counts().sort_index().rename(LABEL_NAMES)
    save_bar(lab_counts, "Train: 6-way label distribution", "label", "01_labels_6way.png")
    save_bar(train["label"].apply(to_binary).value_counts(), "Train: FAKE vs REAL", "label",
             "02_labels_binary.png")
    save_bar(speakers.value_counts().head(15), "Train: top 15 speakers", "", "03_top_speakers.png",
             horizontal=True)
    save_bar(subjects.value_counts().head(15), "Train: top 15 subjects", "", "04_top_subjects.png",
             horizontal=True)
    save_bar(dates.dt.year.value_counts().sort_index(), "Train: claims per year", "year",
             "05_claims_per_year.png")
    print("=" * 60, f"\nCharts saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
