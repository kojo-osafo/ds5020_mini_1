"""
build_matrix.py  (DE1 to DE6)

    python src/build_matrix.py
    python src/build_matrix.py --value binary --min-customers 20

Pipeline
    DE1  profile raw data
    DE2  clean + cleaning log
    DE3  time split + held-out test
    DE4  sparse customer x product matrix from TRAIN only
    DE5  matrix report
    DE6  all of the above in this one script

Outputs
    outputs/de1_data_profile.csv          per-column profile (dtype, missing, unique)
    outputs/de1_sheet_summary.csv         rows and date range per sheet
    outputs/de2_cleaning_log.csv/.md      rows removed by each cleaning rule, in order
    outputs/de3_split_report.md/.json     split + hold-out counts
    outputs/de5_matrix_report.md/.json    shape, nnz, sparsity, memory dense vs sparse
    data/processed/clean_lines.pkl        cleaned invoice lines
    data/processed/A_train.npz            sparse matrix (customers x products)
    data/processed/customer_index.csv     row -> customer id, country   (PRIVATE, git-ignored)
    data/processed/product_index.csv      col -> stock code, description
    data/processed/train_lines.pkl        training invoice lines
    data/processed/test_heldout.csv       held-out (cust_idx, prod_idx) pairs, anonymised

R1: every decision is written next to the code that applies it.
R2: the product filter, index maps and matrix use TRAIN data only.
R4: outputs/ never contains customer IDs. The ID map lives in data/processed/.
R6: no randomness here; the seed is set anyway.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parent.parent     # repo root (parent of src/)
SEED = 42
RAW_DEFAULT = "data/raw/online_retail_II.xlsx"
CACHE = "data/interim/raw_combined.pkl"      # Excel is slow: read it once
PROC = "data/processed"
CLEAN_OUT = f"{PROC}/clean_lines.pkl"
OUT = "outputs"


COLUMN_ALIASES = {
    "InvoiceNo": "Invoice",
    "UnitPrice": "Price",
    "CustomerID": "Customer ID",
    "Customer_ID": "Customer ID",
}
EXPECTED = ["Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "Customer ID", "Country"]

# DECISION: a real product code is exactly 5 digits, optionally followed by
# letters for variants (85123A, 15056BL). This drops POST, M, D, DOT, CRUK,
# BANK CHARGES, TEST001, gift_0001_10, AMAZONFEE, etc.
PRODUCT_CODE = re.compile(r"^\d{5}[A-Z]*$")


# ---------------------------------------------------------------- DE1
def load_raw(path: str, use_cache: bool = True) -> pd.DataFrame:
    """Load both sheets, tag each row with its sheet, normalise column names."""
    if not os.path.exists(path):
        sys.exit(f"[DE1] Raw file not found: {path}\n"
                 f"Put online_retail_II.xlsx in data/raw/ or pass --raw <path>.")
    if use_cache and os.path.exists(CACHE) and \
            os.path.getmtime(CACHE) > os.path.getmtime(path):
        print(f"[DE1] Using cached raw data: {CACHE}")
        return pd.read_pickle(CACHE)

    try:
        import openpyxl  # noqa: F401  (needed by pandas to read .xlsx)
    except ImportError:
        sys.exit("[DE1] Missing package 'openpyxl'. Run: pip install openpyxl")

    print(f"[DE1] Reading {path} (slow, about 1-2 minutes) ...", flush=True)
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for name, sheet in sheets.items():
        sheet = sheet.rename(columns=COLUMN_ALIASES)
        sheet["sheet"] = name
        frames.append(sheet)
    df = pd.concat(frames, ignore_index=True)

    missing_cols = [c for c in EXPECTED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Unexpected columns. Missing {missing_cols}. "
                         f"Found {df.columns.tolist()}")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_pickle(CACHE)
    return df


def profile(df: pd.DataFrame) -> None:
    """DE1: data profile table + per-sheet summary."""
    os.makedirs(OUT, exist_ok=True)

    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })
    prof.index.name = "column"
    prof.to_csv(f"{OUT}/de1_data_profile.csv")

    sheet_summary = (
        df.groupby("sheet")
          .agg(rows=("Invoice", "size"),
               first_date=("InvoiceDate", "min"),
               last_date=("InvoiceDate", "max"))
    )
    sheet_summary.loc["TOTAL"] = [len(df), df["InvoiceDate"].min(),
                                  df["InvoiceDate"].max()]
    sheet_summary.to_csv(f"{OUT}/de1_sheet_summary.csv")

    inv = df["Invoice"].astype(str)
    extras = {
        "total_rows": len(df),
        "distinct_invoices": inv.nunique(),
        "cancellation_lines (Invoice starts with C)": int(inv.str.startswith("C").sum()),
        "lines Quantity <= 0": int((df["Quantity"] <= 0).sum()),
        "lines Price <= 0": int((df["Price"] <= 0).sum()),
        "distinct customers (non-missing)": df["Customer ID"].nunique(),
        "distinct stock codes": df["StockCode"].astype(str).nunique(),
        "exact duplicate lines": int(df.drop(columns="sheet").duplicated().sum()),
    }

    print("\n=== DE1 DATA PROFILE ===")
    print(prof.to_string())
    print("\n--- Rows per sheet ---")
    print(sheet_summary.to_string())
    print("\n--- Quick facts ---")
    for k, v in extras.items():
        print(f"{k:45s} {v:,}")


# ---------------------------------------------------------------- DE2
def clean(df: pd.DataFrame):
    """
    DE2: apply the cleaning rules IN ORDER and log rows removed by each one.
    Counts are sequential: a row is credited to the first rule that removes it.
    """
    log = [("Raw data (start)", 0, len(df), "")]

    def apply(rule_name, keep_mask, reason):
        nonlocal df
        before = len(df)
        df = df[keep_mask(df)]
        log.append((rule_name, before - len(df), len(df), reason))

    # Normalise types first (not a removal step, so not logged as one).
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()

    # 1. Cancellations. Invoice numbers beginning with "C" are cancellations,
    #    and their quantities are negative.
    apply("1. Cancellation invoices (Invoice starts with 'C')",
          lambda d: ~d["Invoice"].str.startswith("C"),
          "A cancelled order is not a purchase.")

    # 2. Non-positive quantity (returns, adjustments, damaged stock).
    apply("2. Quantity <= 0",
          lambda d: d["Quantity"] > 0,
          "Returns/adjustments are not purchases.")

    # 3. Non-positive price (free samples, manual adjustments, bad debt).
    apply("3. Price <= 0",
          lambda d: d["Price"] > 0,
          "Zero/negative price lines are not real sales.")

    # 4. Non-product codes (postage, manual, fees, test rows ...).
    apply("4. Non-product StockCode (not 5 digits + optional letters)",
          lambda d: d["StockCode"].str.match(PRODUCT_CODE),
          "POST, M, D, DOT, BANK CHARGES, gift vouchers etc. are not products.")

    # 5. Missing customer ID. Cannot build a customer x product row without it.
    apply("5. Missing Customer ID",
          lambda d: d["Customer ID"].notna(),
          "No customer row to put the purchase in (~1/5 of raw lines).")

    df["Customer ID"] = df["Customer ID"].astype("int64")

    # Report only (do NOT remove): exact duplicate lines.
    # DECISION: kept. A wholesaler can legitimately scan the same item twice on
    # one invoice, and the matrix sums quantity anyway. Revisit in ML7 if wanted.
    n_dup = int(df.duplicated().sum())
    log.append(("(info) Exact duplicate lines, NOT removed", 0, len(df),
                f"{n_dup:,} duplicates kept; they add quantity, not extra columns."))

    log_df = pd.DataFrame(log, columns=["rule", "rows_removed",
                                        "rows_remaining", "reason"])
    log_df["pct_of_raw_removed"] = (
        log_df["rows_removed"] / log_df.loc[0, "rows_remaining"] * 100
    ).round(2)
    return df.reset_index(drop=True), log_df


def save_clean(df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PROC, exist_ok=True)

    log_df.to_csv(f"{OUT}/de2_cleaning_log.csv", index=False)

    md = ["| Rule | Rows removed | Rows remaining | % of raw | Reason |",
          "|---|---:|---:|---:|---|"]
    for _, r in log_df.iterrows():
        md.append(f"| {r['rule']} | {r['rows_removed']:,} | "
                  f"{r['rows_remaining']:,} | {r['pct_of_raw_removed']} | "
                  f"{r['reason']} |")
    with open(f"{OUT}/de2_cleaning_log.md", "w") as f:
        f.write("\n".join(md) + "\n")

    df[EXPECTED].to_pickle(CLEAN_OUT)

    print("\n=== DE2 CLEANING LOG ===")
    print(log_df[["rule", "rows_removed", "rows_remaining",
                  "pct_of_raw_removed"]].to_string(index=False))
    print(f"\nClean lines: {len(df):,}")
    print(f"Customers:   {df['Customer ID'].nunique():,}")
    print(f"Products:    {df['StockCode'].nunique():,}")
    print(f"Date range:  {df['InvoiceDate'].min()}  ->  {df['InvoiceDate'].max()}")
    print(f"Saved: {CLEAN_OUT}")


# ---------------------------------------------------------------- DE3
def time_split(df: pd.DataFrame, test_months: int = 2):
    """Train = everything before the cutoff, test = the last `test_months`."""
    cutoff = (df["InvoiceDate"].max() - pd.DateOffset(months=test_months)).normalize()
    train = df[df["InvoiceDate"] < cutoff].copy()
    test = df[df["InvoiceDate"] >= cutoff].copy()
    assert train["InvoiceDate"].max() < cutoff <= test["InvoiceDate"].min(), "time leak"
    return train, test, cutoff


# ---------------------------------------------------------------- DE4
def build_train_matrix(train: pd.DataFrame, min_customers: int, value: str):
    """
    Customer x product sparse matrix from TRAIN lines only.

    DECISIONS (R1)
    - Aggregate to one (customer, product) cell first by SUMMING quantity, so
      repeated lines for the same item are one cell, not several.
    - Product filter: keep products bought by >= min_customers distinct train
      customers. Rare products give noisy factors and cannot be learned from.
      Counted on train only (R2).
    - Purchase value: 'log' = log(1 + total quantity) (default; dampens the
      wholesaler who buys 500 units), 'binary' = 1 if ever bought,
      'count' = raw total quantity.
    """
    pairs = (train.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"]
                  .sum())
    n_buyers = pairs.groupby("StockCode")["Customer ID"].nunique()
    keep_codes = n_buyers[n_buyers >= min_customers].index
    pairs = pairs[pairs["StockCode"].isin(keep_codes)]

    cust_ids = np.sort(pairs["Customer ID"].unique())
    prod_codes = np.sort(pairs["StockCode"].unique())
    rows = np.searchsorted(cust_ids, pairs["Customer ID"].to_numpy())
    cols = np.searchsorted(prod_codes, pairs["StockCode"].to_numpy())

    qty = pairs["Quantity"].to_numpy(dtype=np.float64)
    if value == "log":
        vals = np.log1p(qty)
    elif value == "binary":
        vals = np.ones_like(qty)
    elif value == "count":
        vals = qty
    else:
        raise ValueError("value must be log, binary or count")

    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(cust_ids), len(prod_codes)))
    A.sum_duplicates()

    # index maps
    train_k = train[train["StockCode"].isin(prod_codes)]
    desc = (train_k.groupby("StockCode")["Description"]
                   .agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    prod_index = pd.DataFrame({"col": np.arange(len(prod_codes)),
                               "StockCode": prod_codes,
                               "Description": desc.reindex(prod_codes).to_numpy()})
    ctry = (train[train["Customer ID"].isin(cust_ids)]
            .groupby("Customer ID")["Country"]
            .agg(lambda s: s.mode().iat[0]))
    cust_index = pd.DataFrame({"row": np.arange(len(cust_ids)),
                               "Customer ID": cust_ids,
                               "Country": ctry.reindex(cust_ids).to_numpy()})
    stats = {"products_before_filter": int(len(n_buyers)),
             "products_after_filter": int(len(prod_codes)),
             "min_customers": int(min_customers),
             "purchase_value": value}
    return A, cust_index, prod_index, stats


def build_heldout(test: pd.DataFrame, A: sp.csr_matrix,
                  cust_index: pd.DataFrame, prod_index: pd.DataFrame):
    """
    Held-out purchases (R2): for each test customer, the products they bought
    in the test period that they NEVER bought in train.

    DECISIONS (R1)
    - Only customers present in the train matrix are evaluated (a new customer
      has no history to recommend from: cold start, reported as a risk).
    - Only products in the train catalogue count (a model cannot recommend a
      product it has never seen).
    - Products the customer already bought in train are removed, so a hit means
      the model found something NEW for them.
    """
    t = test[["Customer ID", "StockCode"]].drop_duplicates()
    n_pairs = len(t)

    c_ids = cust_index["Customer ID"].to_numpy()
    p_codes = prod_index["StockCode"].to_numpy()

    in_c = t["Customer ID"].isin(c_ids)
    t = t[in_c]
    n_after_cust = len(t)
    in_p = t["StockCode"].isin(p_codes)
    t = t[in_p]
    n_after_prod = len(t)

    r = np.searchsorted(c_ids, t["Customer ID"].to_numpy())
    c = np.searchsorted(p_codes, t["StockCode"].to_numpy())
    already = np.asarray(A[r, c]).ravel() > 0
    held = pd.DataFrame({"cust_idx": r[~already], "prod_idx": c[~already]})

    stats = {"test_lines": int(len(test)),
             "test_unique_customer_product_pairs": int(n_pairs),
             "dropped_customer_not_in_train": int(n_pairs - n_after_cust),
             "dropped_product_not_in_train_catalogue": int(n_after_cust - n_after_prod),
             "dropped_already_bought_in_train": int(already.sum()),
             "heldout_pairs": int(len(held)),
             "test_customers_evaluated": int(held["cust_idx"].nunique()),
             "test_customers_total": int(test["Customer ID"].nunique())}
    return held, stats


# ---------------------------------------------------------------- DE5
def matrix_report(A: sp.csr_matrix) -> dict:
    n, m = A.shape
    dense_mb = n * m * 8 / 1e6
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    return {"n_customers": int(n), "n_products": int(m),
            "non_zeros": int(A.nnz),
            "sparsity_pct": round(100 * (1 - A.nnz / (n * m)), 2),
            "avg_products_per_customer": round(A.nnz / n, 1),
            "dense_memory_MB": round(dense_mb, 1),
            "sparse_memory_MB": round(sparse_mb, 2),
            "dense_over_sparse": round(dense_mb / sparse_mb, 1),
            "k50_factors_MB": round((n + m) * 50 * 8 / 1e6, 2)}


def write_report(name: str, title: str, d: dict) -> None:
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(d, f, indent=2)
    lines = [f"### {title}", "", "| Item | Value |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |"
              for k, v in d.items()]
    with open(f"{OUT}/{name}.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--no-cache", action="store_true",
                    help="re-read the Excel file even if a cache exists")
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--min-customers", type=int, default=10)
    ap.add_argument("--value", choices=["log", "binary", "count"], default="log")
    args = ap.parse_args()

    # Work from the repo root so it runs from any folder; keep a user-supplied
    # --raw path valid by making it absolute first.
    raw_path = os.path.abspath(args.raw) if os.path.exists(args.raw) else args.raw
    os.chdir(ROOT)
    if not os.path.exists(raw_path) and os.path.exists(args.raw):
        raw_path = args.raw

    np.random.seed(SEED)
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    timing = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        timing[name] = round(time.perf_counter() - t0, 2)
        return out

    # DE1 + DE2
    from_cache = (not args.no_cache and os.path.exists(raw_path)
                  and os.path.exists(CACHE)
                  and os.path.getmtime(CACHE) > os.path.getmtime(raw_path))
    raw = timed("raw_load_s", lambda: load_raw(raw_path, use_cache=not args.no_cache))
    timing["raw_load_from_cache"] = bool(from_cache)
    profile(raw)
    clean_df, log_df = timed("clean_s", lambda: clean(raw))
    save_clean(clean_df, log_df)

    # DE3
    train, test, cutoff = timed("split_s",
                                lambda: time_split(clean_df, args.test_months))
    print(f"\n[DE3] cutoff {cutoff.date()} | train lines {len(train):,} "
          f"| test lines {len(test):,}")

    # DE4 (train only)
    A, cust_index, prod_index, fstats = timed(
        "matrix_s", lambda: build_train_matrix(train, args.min_customers, args.value))
    held, hstats = timed("heldout_s",
                         lambda: build_heldout(test, A, cust_index, prod_index))

    sp.save_npz(f"{PROC}/A_train.npz", A)
    cust_index.to_csv(f"{PROC}/customer_index.csv", index=False)
    prod_index.to_csv(f"{PROC}/product_index.csv", index=False)
    train.to_pickle(f"{PROC}/train_lines.pkl")
    held.to_csv(f"{PROC}/test_heldout.csv", index=False)

    split = {"cutoff_date": str(cutoff.date()),
             "train_lines": int(len(train)), **fstats, **hstats}
    write_report("de3_split_report", "DE3 split and hold-out report", split)

    # DE5
    write_report("de5_matrix_report", "DE5 matrix report", matrix_report(A))

    # Timing. A nightly run repeats cleaning + matrix building. Reading the
    # Excel file, the time split and the held-out test set are one-off
    # evaluation work, so they are reported but NOT counted in matrix_rebuild_s.
    timing["matrix_rebuild_s"] = round(timing["clean_s"] + timing["matrix_s"], 2)
    with open(f"{OUT}/build_timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print("\n### Build timing (seconds)")
    for k, v in timing.items():
        print(f"  {k:22s} {v}")
    print(f"\nDone. Matrix saved to {PROC}/A_train.npz")


if __name__ == "__main__":
    main()
