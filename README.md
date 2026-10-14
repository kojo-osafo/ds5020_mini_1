# Customers Like You (GiftLane): PCA or SVD

Team: Kwadwo Osafo   |   Raw data row count : 1,067,371
Raw Data Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

## J1. Success targets 
| # | Target | Result |
|---|---|---|
| 1 | Compressed factors under 10 MB | see `outputs/results.json` |
| 2 | Nightly job (fit + score all customers) under 60 s | ~5secs fit time + loading matrix, ~40secs build time from scratch (without cache used via `--no-cache` flag with `build_matrix.py`)|
| 3 | Hit rate at 10 beats the best-seller baseline, with 95% interval above 0 | Compressed model honestly loses to the baseline |
| 4 | 3 to 6 customer groups that marketing can name | 5 core groups identified; see `outputs/ml6_segments.csv`|

## How to run (R6)
First, ensure the raw data has been downloaded from the link above, and stored in data/raw/ in the root.
```bash
pip install -r requirements.txt
python src/build_matrix.py        # DE1-DE6: profile, clean, split, sparse matrix, reports
python src/ml_analysis.py         # ML1-ML7: k sweep, recommender, evaluation, PCA vs SVD, segments
```
Seed is fixed at 42. `notebooks/analysis.ipynb` runs the same steps top to bottom and shows the results.

## Layout
`src/` code | `data/` raw and processed data (git-ignored; contains customer IDs, never share)
| `outputs/` tables, figures, memo, slides (no customer IDs) | `decision_log.md` | `contribution_log.md`

## Roles
As the sole engineer, I shouldered all roles and responsibilities for this project.


## Renaming customer groups
Edit `outputs/segment_names.json`, e.g. `{"0": "UK bulk wholesalers", "1": "..."}`, then rerun
`ml_analysis.py`, `make_memo.py` and `make_slides.py`.
