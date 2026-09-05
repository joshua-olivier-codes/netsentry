from pathlib import Path

import pandas as pd

# =========================================================
# NETSENTRY - DATA PREPROCESSING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "network_traffic.csv"


# =========================================================
# FIND DATASET
# =========================================================


def find_csv_files():
    files = sorted(RAW_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found under:\n{RAW_DIR}")

    return files


# =========================================================
# CLEAN DATAFRAME
# =========================================================


def clean_dataframe(df):

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Find label column
    label_column = None

    for column in df.columns:

        if column.lower() == "label":
            label_column = column
            break

    if label_column is None:

        raise ValueError("Could not find Label column.")

    # Clean labels
    df[label_column] = df[label_column].astype(str).str.strip()

    # Remove empty labels
    df = df[df[label_column] != ""].copy()

    # Create normalized target
    df["target"] = df[label_column].apply(
        lambda x: "BENIGN" if x.upper() == "BENIGN" else "ATTACK"
    )

    # Create attack type
    df["attack_type"] = df[label_column].astype(str).str.strip()

    return df


# =========================================================
# MAIN
# =========================================================


def main():

    print("=" * 70)
    print("NETSENTRY - DATA PREPROCESSING")
    print("=" * 70)

    files = find_csv_files()

    print(f"\nFound {len(files)} CSV files.")

    frames = []

    for file in files:

        print(f"\nLoading: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        print(f"  Original rows: {len(df):,}")

        df = clean_dataframe(df)

        print(f"  Clean rows:    {len(df):,}")

        frames.append(df)

    print("\nCombining datasets...")

    combined = pd.concat(frames, ignore_index=True)

    print(f"\nTotal rows: " f"{len(combined):,}")

    print(f"Total columns: " f"{len(combined.columns)}")

    # -----------------------------------------------------
    # Binary distribution
    # -----------------------------------------------------

    print("\nBinary distribution:")

    print(combined["target"].value_counts())

    # -----------------------------------------------------
    # Attack distribution
    # -----------------------------------------------------

    print("\nAttack types:")

    print(combined["attack_type"].value_counts())

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving processed dataset...")

    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to:\n{OUTPUT_FILE}")

    print(f"\nFinal dataset shape:")

    print(combined.shape)

    print("\n" + "=" * 70)

    print("PREPROCESSING COMPLETE")

    print("=" * 70)


if __name__ == "__main__":
    main()
