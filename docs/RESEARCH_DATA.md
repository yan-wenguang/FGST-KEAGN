# Processed research data

These five CSV files are copied byte-for-byte from the supplied `new_data.zip`. They are real processed research inputs, separate from `examples/synthetic.csv`.

| File | Rows | Columns | Label counts |
|---|---:|---:|---|
| [fire_preprocessed_data.csv](../data/processed/fire_preprocessed_data.csv) | 1215 | 24 | {'1': 631, '-1': 584} |
| [algeria_wildfire_enriched_for_pipeline.csv](../data/processed/algeria_wildfire_enriched_for_pipeline.csv) | 517 | 25 | {'0': 247, '1': 270} |
| [algeria_wildfire_for_pipeline.csv](../data/processed/algeria_wildfire_for_pipeline.csv) | 517 | 15 | {'0': 247, '1': 270} |
| [america_wildfire_for_pipeline.csv](../data/processed/america_wildfire_for_pipeline.csv) | 64196 | 22 | {'1': 32098, '0': 32098} |
| [america_wildfire_horizon_for_pipeline.csv](../data/processed/america_wildfire_horizon_for_pipeline.csv) | 64196 | 24 | {'1': 32098, '0': 32098} |

The target is `fire_occurred`. Exact field lists, label distributions and SHA-256 checksums are in [manifest.json](../data/processed/manifest.json). All five files were scanned for blank, nonnumeric and nonfinite values; results are recorded in that manifest. This checks serialization, not scientific correctness or label quality.

The `america_*` files include latitude/longitude, and the `algeria_*` files include X/Y fields. They have not been de-identified or coordinate-redacted. The original filenames are preserved as supplied; geographic source, publisher, licensing, label construction and preprocessing provenance were not verified from filenames. Do not infer a dataset identity or reuse permission solely from a file name.

Raw source files, the location-enriched source CSV, trained weights, row-level predictions and experiment artifacts are not included in this data release. The two `algeria_*` files and two `america_*` files are variants, not independent datasets to concatenate automatically.

## Execute on the primary processed input

```bash
python -m src.pipelines.run_full_pipeline --config configs/processed_primary.yaml
```

This configuration connects the primary processed CSV to the released two-stage implementation. It uses three epochs as an execution smoke check and fallback knowledge priors. It is not the private final paper configuration and does not establish reproduction of published metrics. Increase epochs and supply documented knowledge records for a separately specified experiment.
