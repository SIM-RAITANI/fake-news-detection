"""Step 1a: download the LIAR2 dataset from Hugging Face and save it as CSV files.

Run:  python src/download_data.py
Output: data/raw/train.csv, data/raw/validation.csv, data/raw/test.csv
"""
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("chengxuphd/liar2")

    print("Columns:", ds["train"].column_names)
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        out = RAW_DIR / f"{split}.csv"
        df.to_csv(out, index=False)
        print(f"{split:<10} {len(df):>6} rows  ->  {out}")


if __name__ == "__main__":
    main()
