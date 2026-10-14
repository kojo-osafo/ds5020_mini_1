| Rule | Rows removed | Rows remaining | % of raw | Reason |
|---|---:|---:|---:|---|
| Raw data (start) | 0 | 1,067,371 | 0.0 |  |
| 1. Cancellation invoices (Invoice starts with 'C') | 19,494 | 1,047,877 | 1.83 | A cancelled order is not a purchase. |
| 2. Quantity <= 0 | 3,457 | 1,044,420 | 0.32 | Returns/adjustments are not purchases. |
| 3. Price <= 0 | 2,750 | 1,041,670 | 0.26 | Zero/negative price lines are not real sales. |
| 4. Non-product StockCode (not 5 digits + optional letters) | 4,793 | 1,036,877 | 0.45 | POST, M, D, DOT, BANK CHARGES, gift vouchers etc. are not products. |
| 5. Missing Customer ID | 234,245 | 802,632 | 21.95 | No customer row to put the purchase in (~1/5 of raw lines). |
| (info) Exact duplicate lines, NOT removed | 0 | 802,632 | 0.0 | 11,930 duplicates kept; they add quantity, not extra columns. |
