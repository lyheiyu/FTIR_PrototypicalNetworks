# Data-access conditions

This is a memory-matched incremental-head comparison. It is not a comparison that equalizes all historical learning opportunities across learned embeddings and raw spectra.

| Stage | M0/M1 | Raw-spectrum SVM |
| --- | --- | --- |
| Historical encoder training | Original four-class base_train, supervised CE; original base_validation used for checkpoint selection | No learned encoder or historical representation training |
| SVM C selection | Reuse original architecture-run base-validation episode selection, separately for M0/M1 and each shot | Reuse corresponding original raw-spectrum base-validation selection |
| Session 0 head fitting | Same fixed 20 archived base_train exemplars per base class | Exactly the same 80 indices |
| Session 1 head update | Retained 80 base exemplars plus K support for the first novel class | Exactly the same 80+K indices |
| Session 2 head update | Retain all previous support, add K for the second novel class | Exactly the same 80+2K indices |
| Novel query use | Frozen inference and scoring only; no gradient, C tuning or support replacement | Scoring only; no C tuning or support replacement |
| Base query use | Sample archived base_validation; these labels already informed historical model/C selection | Same base query indices; these labels already informed C selection |

M0 and M1 have matched historical training data and budgets. The raw-spectrum SVM reference has matching retained exemplar access during the incremental phase, but it did not learn an encoder from all historical base training labels. Do not describe this as equal total supervised-data access for all methods. Nor is it an all-base-data replay SVM baseline: such a classifier would be allowed more than the 20 retained base samples/class and is not evaluated here.

The term full SVM refit means a fresh multiclass linear SVM on the entire currently retained seen-class library. It is not an incremental/partial SVM update, a fit restricted to only the arriving class, or a refit on every historical raw spectrum. Local class-bank registration and this complete head refit are both measured, separately, and their serving-update sum is reported. Query feature extraction and row-local preprocessing are excluded from these registration/refit timings.

No encoder training occurs in this evaluation. Historical encoder-training cost remains in the parent study's costs.csv and is not included in the incremental update timings. Checkpoint and array hashes, original role manifests, and archived episode identities permit an audit of the conditions above. The absence of independently untouched base test samples and verified physical specimen IDs limits all-seen conclusions to exploratory evidence.
