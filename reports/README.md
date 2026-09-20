# Shared reports

This folder is the portable, curated result bundle for the notebooks and slides.
It contains aggregate metrics, compact fold summaries and four selected figures
(PNG for display, PDF for publication). There are no participant trials, full
stimulus texts, word-level prediction tables, model weights or activation arrays.

`manifest.json` records repository-relative source names and SHA-256 hashes.
Verify without the original datasets using `bash run.sh verify-reports`.
Regenerate from the full local runs using `bash run.sh export-reports`.
The exporter accepts only its explicit allowlist and checks source manifests.

Subfolders: `baseline/`, `sae_fidelity/`, `sae_prediction/`,
`scaling_sensitivity/`, and `tables/`. The table workshop generates the last
folder separately, with its own relative-path manifests.

A report snapshot is sufficient to read, plot and export the findings. It is
not sufficient to refit the models or verify every underlying observation;
those tasks require the original data and complete local experiment artifacts.
