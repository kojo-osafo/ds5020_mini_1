"""
ml_analysis.py  -  ML1 to ML7 in one reproducible script.

Run AFTER build_matrix.py:
    python src/ml_analysis.py

ML1  decision note
ML2  k sweep: reconstruction error + validation hit rate, choose k
ML3  recommender: product-product cosine similarity on factor vectors
ML4  hit rate at 10 on held-out purchases vs best-seller baseline, nightly timing
ML5  PCA vs Truncated SVD at the same k (comparison table)
ML6  customer map: cluster customers into 3-6 groups, describe them
ML7  sensitivity of hit rate to purchase value (log / binary / count)

R2: k is chosen on a VALIDATION split carved out of the training matrix.
R4: no customer IDs in any output file or figure.
"""
import argparse
import json
import os
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

SEED = 42
PROC, OUT = "data/processed", "outputs"
TOPN = 10
K_GRID = [10, 20, 50, 100, 200, 300, 500]


# ------------------------------------------------------------ utilities
def measure(fn):
    """Run fn(); return (result, seconds, peak extra memory in MB)."""
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.01)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    return out, dt, (peak[0] - base) / 1e6


def fro2(A):
    return float(A.multiply(A).sum())


def fit_svd(A, k):
    return TruncatedSVD(n_components=k, random_state=SEED, n_iter=7).fit(A)


def svd_energy_kept(svd, A):
    return float((svd.singular_values_ ** 2).sum() / fro2(A))


def fit_pca(A, k):
    """PCA with implicit centring on sparse input (scikit-learn >= 1.4)."""
    try:
        return PCA(n_components=k, svd_solver="arpack", random_state=SEED).fit(A), ""
    except Exception as e:                      # fallback: densify (slow, big)
        pca = PCA(n_components=k, svd_solver="randomized",
                  random_state=SEED).fit(A.toarray())
        return pca, f"dense fallback ({type(e).__name__})"


def pca_scores(pca, A):
    """Customer scores for PCA on sparse A without densifying."""
    return np.asarray(A @ pca.components_.T) - pca.mean_ @ pca.components_.T


def pca_rel_err(pca, A):
    n = A.shape[0]
    ac2 = fro2(A) - n * float((pca.mean_ ** 2).sum())        # ||A centred||^2
    kept = float((pca.explained_variance_).sum() * (n - 1))   # sum of s_i^2
    return float(np.sqrt(max(ac2 - kept, 0.0) / fro2(A)))


# ------------------------------------------------------------ ML3 recommender
def product_vectors(components):
    return normalize(components.T)                   # one unit row per product


def make_cf_scorer(A, P):
    """Score(customer, product) = sum over bought items of value * cosine sim."""
    S = (P @ P.T).astype(np.float32)
    A32 = A.astype(np.float32).tocsr()
    return lambda rows: np.asarray(A32[rows] @ S)


def make_pop_scorer(A):
    pop = np.asarray(A.getnnz(axis=0), dtype=np.float32)        # distinct buyers
    return lambda rows: np.tile(pop, (len(rows), 1))


def similar_products(P, j, n=10):
    s = P @ P[j]
    s[j] = -np.inf
    return np.argsort(-s)[:n]


def recommend_for_customer(score_fn, A, i, n=10):
    sc = score_fn(np.array([i]))[0].copy()
    sc[A[i].indices] = -np.inf
    return np.argsort(-sc)[:n]


def hits_at_n(score_fn, A_hist, H, rows, n=TOPN, batch=1000):
    """Boolean array: for each row, is any held-out product in the top n?"""
    out = np.zeros(len(rows), dtype=bool)
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch]
        sc = np.array(score_fn(r), dtype=np.float32)
        br, bc = A_hist[r].nonzero()
        sc[br, bc] = -np.inf                       # never recommend owned items
        top = np.argpartition(-sc, n - 1, axis=1)[:, :n]
        Hd = H[r].toarray()
        out[s:s + batch] = np.take_along_axis(Hd, top, axis=1).sum(axis=1) > 0
    return out


def top_n_all(score_fn, A, n=TOPN, batch=1000):
    """The nightly scoring step: top-n for EVERY customer."""
    rows_all = np.arange(A.shape[0])
    res = np.zeros((A.shape[0], n), dtype=np.int32)
    for s in range(0, A.shape[0], batch):
        r = rows_all[s:s + batch]
        sc = np.array(score_fn(r), dtype=np.float32)
        br, bc = A[r].nonzero()
        sc[br, bc] = -np.inf
        res[s:s + batch] = np.argpartition(-sc, n - 1, axis=1)[:, :n]
    return res


def nightly_job(A, k):
    svd = fit_svd(A, k)
    P = product_vectors(svd.components_)
    top_n_all(make_cf_scorer(A, P), A)


def bootstrap_diff(h_a, h_b, reps=2000):
    rng = np.random.default_rng(SEED)
    d = h_a.astype(float) - h_b.astype(float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------ ML2 validation split
def make_validation(A, cust_frac=0.25, hide_frac=0.2, min_nnz=5):
    """Hide ~20% of the purchases of 25% of customers (training data only)."""
    rng = np.random.default_rng(SEED)
    nnz_row = np.diff(A.indptr)
    eligible = np.where(nnz_row >= min_nnz)[0]
    chosen = rng.choice(eligible, size=max(1, int(len(eligible) * cust_frac)),
                        replace=False)
    hr, hc = [], []
    for i in chosen:
        cols = A.indices[A.indptr[i]:A.indptr[i + 1]]
        n_hide = max(1, int(round(hide_frac * len(cols))))
        hide = rng.choice(cols, n_hide, replace=False)
        hr += [i] * n_hide
        hc += list(hide)
    H = sp.csr_matrix((np.ones(len(hr)), (hr, hc)), shape=A.shape)
    A_tr = (A - A.multiply(H)).tocsr()
    A_tr.eliminate_zeros()
    return A_tr, H, np.sort(chosen)


# ------------------------------------------------------------ ML5 component 1
def describe_comp1(method, comp, z1, A, prod_desc):
    same_sign = max((comp > 0).mean(), (comp < 0).mean())
    sign = 1 if (comp > 0).mean() >= 0.5 else -1
    n_prod = np.asarray(A.getnnz(axis=1)).astype(float)
    total = np.asarray(A.sum(axis=1)).ravel()
    r_n = abs(np.corrcoef(z1, n_prod)[0, 1]) if z1.std() > 0 else 0.0
    r_t = abs(np.corrcoef(z1, total)[0, 1]) if z1.std() > 0 else 0.0
    order = np.argsort(-sign * comp)
    top_pos = [prod_desc[j] for j in order[:4]]
    top_neg = [prod_desc[j] for j in order[::-1][:4]]
    if method == "SVD" and same_sign > 0.9 and max(r_n, r_t) > 0.6:
        text = (f"Overall purchase volume: {same_sign:.0%} of products load with "
                f"the same sign; score tracks number of products bought "
                f"(r={r_n:.2f}) and total value (r={r_t:.2f}).")
    elif method == "PCA":
        text = (f"Main way customers differ from the average customer. "
                f"Pushes up: {'; '.join(top_pos)}. Pushes down: {'; '.join(top_neg)}.")
    else:
        text = (f"Mixed pattern ({same_sign:.0%} same-sign loadings; r with number "
                f"of products {r_n:.2f}). Top products: {'; '.join(top_pos)}.")
    return text


# ------------------------------------------------------------ ML6 segments
def build_segments(A, Z, cust_index, prod_desc, names_override):
    X = normalize(Z)                              # direction only: removes volume
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    best = None
    scores = {}
    for kk in range(3, 7):
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(X)
        sil = silhouette_score(X[samp], km.labels_[samp])
        scores[kk] = float(sil)
        if best is None or sil > best[0]:
            best = (sil, kk, km.labels_)
    _, kk, labels = best

    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    xy = Xc @ Vt[:2].T                            # 2-D projection for the figure

    glob_mean = np.asarray(A.mean(axis=0)).ravel()
    n_prod = np.asarray(A.getnnz(axis=1))
    spend = np.asarray(A.sum(axis=1)).ravel()
    med_all = np.median(n_prod)
    rows = []
    for c in range(kk):
        m = labels == c
        lift = np.asarray(A[m].mean(axis=0)).ravel() - glob_mean
        top = np.argsort(-lift)[:5]
        ctry = cust_index.loc[m, "Country"].value_counts(normalize=True)
        top_ctry = ", ".join(f"{k} {v:.0%}" for k, v in ctry.head(3).items())
        uk = ctry.get("United Kingdom", 0.0)
        med = np.median(n_prod[m])
        tier = "Heavy" if med > 1.3 * med_all else "Light" if med < 0.7 * med_all else "Core"
        geo = "UK" if uk >= 0.9 else "international-leaning" if uk < 0.7 else "mostly-UK"
        kw = str(prod_desc[top[0]]).title()[:28]
        draft = f"{tier} {geo} buyers: {kw} fans"
        rows.append({"segment": c,
                     "name": names_override.get(str(c), draft),
                     "draft_name": draft,
                     "customers": int(m.sum()),
                     "share_pct": round(100 * m.mean(), 1),
                     "median_products_bought": float(med),
                     "median_total_value": round(float(np.median(spend[m])), 2),
                     "top_countries": top_ctry,
                     "top_products": " | ".join(str(prod_desc[j]) for j in top)})
    return pd.DataFrame(rows), labels, xy, scores


# ------------------------------------------------------------ main
ML1_TEXT = """# ML1 Decision note
**Main model: Truncated SVD.  Comparison model: PCA (ML5).**
- The purchase matrix is large and sparse. Truncated SVD works on it directly;
  PCA centres the columns, which turns the zeros into small negative numbers and
  makes the matrix dense (or forces special solvers).
- A recommender needs customer factors AND product factors from one fit. SVD
  gives both (U S and V).
- Without centring, SVD describes what customers actually bought. PCA describes
  how they differ from the average customer, which is not the same question as
  "what do customers like you also buy".
- Hypothesis to test in ML5: both methods give similar hit rates, but SVD is
  faster / lighter, and PCA's first component is more interpretable.

Author: Kwadwo Osafo
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ml7", action="store_true")
    args = ap.parse_args()
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    # ML1
    note = f"{OUT}/ml1_decision_note.md"
    if not os.path.exists(note):
        open(note, "w").write(ML1_TEXT)
        print(f"[ML1] wrote {note}")

    # load
    A = sp.load_npz(f"{PROC}/A_train.npz").tocsr()
    cust_index = pd.read_csv(f"{PROC}/customer_index.csv")
    prod_index = pd.read_csv(f"{PROC}/product_index.csv")
    prod_desc = prod_index["Description"].fillna("").to_numpy()
    held = pd.read_csv(f"{PROC}/test_heldout.csv")
    H_test = sp.csr_matrix((np.ones(len(held)), (held.cust_idx, held.prod_idx)),
                           shape=A.shape)
    H_test.data[:] = 1
    test_rows = np.sort(held["cust_idx"].unique())
    n, m = A.shape
    kmax = min(n, m) - 1
    grid = [k for k in K_GRID if k <= kmax] or [max(2, kmax // 2)]
    R = {"shape": [n, m], "nnz": int(A.nnz), "test_customers": int(len(test_rows)),
         "heldout_pairs": int(len(held))}

    # ML2 ---------------------------------------------------------
    A_tr, H_val, val_rows = make_validation(A)
    pop_val = hits_at_n(make_pop_scorer(A_tr), A_tr, H_val, val_rows).mean()
    sweep = []
    for k in grid:
        svd_full, t_fit, _ = measure(lambda: fit_svd(A, k))
        err = float(np.sqrt(max(0, 1 - svd_energy_kept(svd_full, A))))
        svd_v = fit_svd(A_tr, k)
        hv = hits_at_n(make_cf_scorer(A_tr, product_vectors(svd_v.components_)),
                       A_tr, H_val, val_rows).mean()
        sweep.append({"k": k, "rel_recon_error": round(err, 4),
                      "energy_kept": round(1 - err ** 2, 4),
                      "val_hit_rate_at_10": round(float(hv), 4),
                      "val_baseline_hit_rate": round(float(pop_val), 4),
                      "fit_seconds": round(t_fit, 2)})
        print(f"[ML2] k={k:4d} err={err:.3f} val_hit={hv:.3f} fit={t_fit:.1f}s")
    sw = pd.DataFrame(sweep)
    sw.to_csv(f"{OUT}/ml2_k_sweep.csv", index=False)
    best_val = sw["val_hit_rate_at_10"].max()
    k_star = int(sw.loc[sw["val_hit_rate_at_10"] >= 0.98 * best_val, "k"].min())
    R["k_chosen"] = k_star
    R["k_rule"] = "smallest k within 2% of the best validation hit rate"

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(sw.k, sw.rel_recon_error, "o-", color="#2a6f97")
    ax[0].set(xlabel="k (components)", ylabel="relative reconstruction error",
              title="What compression loses")
    ax[1].plot(sw.k, sw.val_hit_rate_at_10, "o-", color="#2a6f97", label="SVD")
    ax[1].axhline(pop_val, ls="--", color="grey", label="best sellers")
    ax[1].axvline(k_star, ls=":", color="#d62828", label=f"chosen k={k_star}")
    ax[1].set(xlabel="k (components)", ylabel="validation hit rate at 10",
              title="What compression buys")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ml2_k_sweep.png", dpi=160)
    plt.close(fig)

    # ML4 + ML5: final fits on the full training matrix ---------------
    svd, t_svd, mem_svd = measure(lambda: fit_svd(A, k_star))
    (pca, pca_note), t_pca, mem_pca = measure(lambda: fit_pca(A, k_star))

    Z_svd = svd.transform(A)
    Z_pca = pca_scores(pca, A)
    P_svd, P_pca = product_vectors(svd.components_), product_vectors(pca.components_)

    h_svd = hits_at_n(make_cf_scorer(A, P_svd), A, H_test, test_rows)
    h_pca = hits_at_n(make_cf_scorer(A, P_pca), A, H_test, test_rows)
    h_pop = hits_at_n(make_pop_scorer(A), A, H_test, test_rows)
    ci = bootstrap_diff(h_svd, h_pop)

    _, t_night, mem_night = measure(lambda: nightly_job(A, k_star))
    P = P_svd
    S_mb = (P @ P.T).astype(np.float32).nbytes / 1e6
    _, t_score_all, _ = measure(
        lambda: top_n_all(make_cf_scorer(A, P_svd), A))

    dense_mb = n * m * 8 / 1e6
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    svd_mb = (Z_svd.nbytes + svd.components_.nbytes) / 1e6
    pca_mb = (Z_pca.nbytes + pca.components_.nbytes + pca.mean_.nbytes) / 1e6

    R.update({
        "dense_MB": round(dense_mb, 1), "sparse_MB": round(sparse_mb, 2),
        "svd_factors_MB": round(svd_mb, 2), "pca_factors_MB": round(pca_mb, 2),
        "similarity_matrix_MB": round(S_mb, 1),
        "svd_hit": round(float(h_svd.mean()), 4),
        "pca_hit": round(float(h_pca.mean()), 4),
        "baseline_hit": round(float(h_pop.mean()), 4),
        "svd_minus_baseline_95ci": [round(ci[0], 4), round(ci[1], 4)],
        "svd_beats_baseline": bool(ci[0] > 0),
        "svd_fit_s": round(t_svd, 2), "pca_fit_s": round(t_pca, 2),
        "svd_peak_MB": round(mem_svd, 1), "pca_peak_MB": round(mem_pca, 1),
        "nightly_job_s": round(t_night, 2), "nightly_job_peak_MB": round(mem_night, 1),
        "score_all_customers_s": round(t_score_all, 2),
        "svd_rel_error": round(float(np.sqrt(max(0, 1 - svd_energy_kept(svd, A)))), 4),
        "svd_energy_kept": round(svd_energy_kept(svd, A), 4),
        "pca_var_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "pca_rel_error": round(pca_rel_err(pca, A), 4),
        "pca_note": pca_note,
    })

    # matrix rebuild time comes from build_matrix.py (outputs/build_timing.json)
    tpath = f"{OUT}/build_timing.json"
    if os.path.exists(tpath):
        bt = json.load(open(tpath))
        R["build_timing"] = bt
        R["matrix_rebuild_s"] = bt["matrix_rebuild_s"]
        R["nightly_total_s"] = round(bt["matrix_rebuild_s"] + R["nightly_job_s"], 2)
    else:
        print("[WARN] outputs/build_timing.json missing: run build_matrix.py first")

    prod_desc_s = [str(x)[:30] for x in prod_desc]
    c1_svd = describe_comp1("SVD", svd.components_[0], Z_svd[:, 0], A, prod_desc_s)
    c1_pca = describe_comp1("PCA", pca.components_[0], Z_pca[:, 0], A, prod_desc_s)

    table = pd.DataFrame({
        "Measure": ["k used", "Fit time (seconds)", "Peak memory during fit (MB)",
                    "Size of stored factors (MB)",
                    "Variance explained / relative reconstruction error",
                    "Hit rate at 10 on held-out purchases",
                    "Best-seller baseline hit rate at 10",
                    "What component 1 represents (auto-draft: rewrite in your words)"],
        "PCA": [k_star, R["pca_fit_s"], R["pca_peak_MB"], R["pca_factors_MB"],
                f"{R['pca_var_explained']:.1%} variance explained; "
                f"rel. error {R['pca_rel_error']:.3f}",
                f"{R['pca_hit']:.1%}", f"{R['baseline_hit']:.1%}", c1_pca],
        "Truncated SVD": [k_star, R["svd_fit_s"], R["svd_peak_MB"], R["svd_factors_MB"],
                          f"rel. error {R['svd_rel_error']:.3f} "
                          f"({R['svd_energy_kept']:.1%} of energy kept)",
                          f"{R['svd_hit']:.1%}", f"{R['baseline_hit']:.1%}", c1_svd]})
    table.to_csv(f"{OUT}/ml5_comparison_table.csv", index=False)
    with open(f"{OUT}/ml5_comparison_table.md", "w") as f:
        f.write("| Measure | PCA | Truncated SVD |\n|---|---|---|\n")
        for _, r in table.iterrows():
            f.write(f"| {r['Measure']} | {r['PCA']} | {r['Truncated SVD']} |\n")
    pd.DataFrame({"model": ["Truncated SVD", "PCA", "Best sellers"],
                  "hit_rate_at_10": [R["svd_hit"], R["pca_hit"], R["baseline_hit"]]}
                 ).to_csv(f"{OUT}/ml4_results.csv", index=False)
    print(table.to_string(index=False))

    # ML3 example output for the product page (no customer data)
    pop_order = np.argsort(-np.asarray(A.getnnz(axis=0)))[:5]
    ex = []
    for j in pop_order:
        for rank, s in enumerate(similar_products(P_svd, j), 1):
            ex.append({"product": prod_desc[j], "rank": rank, "also_bought": prod_desc[s]})
    pd.DataFrame(ex).to_csv(f"{OUT}/ml3_similar_products_examples.csv", index=False)

    # ML6 -----------------------------------------------------------
    names_file = f"{OUT}/segment_names.json"
    override = json.load(open(names_file)) if os.path.exists(names_file) else {}
    seg, labels, xy, sil = build_segments(A, Z_svd, cust_index, prod_desc_s, override)
    seg.to_csv(f"{OUT}/ml6_segments.csv", index=False)
    R["segment_silhouette"] = sil
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = plt.get_cmap("tab10")
    for c in seg["segment"]:
        mk = labels == c
        ax.scatter(xy[mk, 0], xy[mk, 1], s=8, alpha=0.55, color=cmap(int(c)),
                   label=f"{c}: {seg.loc[seg.segment == c, 'name'].iloc[0]} "
                         f"({mk.mean():.0%})")
    ax.set(xlabel="map axis 1", ylabel="map axis 2",
           title="GiftLane customer groups (each dot is one customer)")
    ax.legend(fontsize=7, loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ml6_customer_map.png", dpi=160)
    plt.close(fig)

    # ML7 -----------------------------------------------------------
    if not args.skip_ml7:
        import build_matrix as bm
        split = json.load(open(f"{OUT}/de3_split_report.json"))
        train = pd.read_pickle(f"{PROC}/train_lines.pkl")
        rows7 = []
        for v in ["log", "binary", "count"]:
            Av, ci_, pi_, _ = bm.build_train_matrix(train, split["min_customers"], v)
            assert Av.shape == A.shape, "index mismatch between purchase values"
            sv = fit_svd(Av, k_star)
            hv = hits_at_n(make_cf_scorer(Av, product_vectors(sv.components_)),
                           Av, H_test, test_rows)
            rows7.append({"purchase_value": v, "hit_rate_at_10": round(float(hv.mean()), 4),
                          "baseline_hit_rate": R["baseline_hit"]})
            print(f"[ML7] {v}: {hv.mean():.3f}")
        pd.DataFrame(rows7).to_csv(f"{OUT}/ml7_sensitivity.csv", index=False)
        R["ml7"] = rows7

    json.dump(R, open(f"{OUT}/results.json", "w"), indent=2)
    print(f"\nk={k_star} | SVD {R['svd_hit']:.1%} | PCA {R['pca_hit']:.1%} | "
          f"best sellers {R['baseline_hit']:.1%} | nightly job {R['nightly_job_s']} s "
          f"| matrix rebuild {R.get('matrix_rebuild_s', 'n/a')} s "
          f"| total {R.get('nightly_total_s', 'n/a')} s")


if __name__ == "__main__":
    main()
