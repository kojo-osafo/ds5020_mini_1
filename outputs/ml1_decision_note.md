# ML1 Decision note
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
