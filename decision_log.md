# Decision log (R1): every decision with its reason

| # | Day | Owner | Decision | Reason / evidence |
|---|---|---|---|---|
| 1 | 2 | DE | Drop invoices starting with "C" | Cancellations are not purchases |
| 2 | 2 | DE | Drop Quantity <= 0 and Price <= 0 | Returns, adjustments, free/bad-debt lines |
| 3 | 2 | DE | Keep only codes matching 5 digits + optional letters | Drops POST, M, D, BANK CHARGES etc. |
| 4 | 2 | DE | Drop lines without Customer ID (~1/5) | No row to place the purchase in |
| 5 | 2 | DE | Keep exact duplicate lines | Genuine repeat scans; quantities are summed anyway |
| 6 | 3 | DE | Test period = last 2 months | Required by brief; time-based, no leakage |
| 7 | 3 | DE | Evaluate only customers/products seen in train; hold out only NEW products | Cold start cannot be tested; a hit must mean something new was found |
| 8 | 3 | DE | Purchase value = log(1 + summed quantity) | Dampens bulk buyers; ML7 compares with binary and count |
| 9 | 3 | DE | Product filter: bought by >= (10) train customers | Rare products give noisy factors; counted on train only |
| 10 | 4 | ML | Main model: Truncated SVD | See `outputs/ml1_decision_note.md` (written before fitting) |
| 11 | 4 | ML | k chosen on a validation split inside train, rule: smallest k within 2% of best | Keeps the test set untouched (R2) |
| 12 | 5 | ML | Customer map clusters on row-normalised factors | Removes the "volume" component so groups reflect taste |
| 13 | 5 | ML | Number of segments chosen by silhouette over 3 to 6 | Objective; check names make sense to marketing |
