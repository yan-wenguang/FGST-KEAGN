# Synthetic Data Card

## Origin and intended use

`examples/synthetic.csv` is an independently generated fixture for public utility testing and a conventional classification illustration. The generator uses Python's standard-library `random.Random(91357)` and does not read real data, private statistics, study configurations or model files. It does not access external services.

The fixture is not a forest-fire dataset, an anonymized study dataset, or evidence of risk-prediction performance. The generic column names deliberately do not encode private study fields.

## Fields and generation

| Field | Domain | Role |
|---|---|---|
| `feature_a`, `feature_b`, `feature_c` | Independent uniform draws in [-1, 1] | Baseline inputs |
| `context_level` | Uniform draw in [0, 1] | Schema context illustration; excluded from model inputs |
| `label` | 0 or 1 | Synthetic target |

For each row, the generator draws the three inputs, context and independent noise in that order. Noise is uniform in [-0.6, 0.6]. The target is:

```text
label = int(1.4*feature_a - 0.9*feature_b + 0.5*feature_c + noise >= 0)
```

The label is calculated before feature serialization rounds values. The noise is not stored as a feature. All four stored real-valued columns use six decimal places; CSV encoding is UTF-8 with an LF line ending.

## Identity and integrity

- Rows: 240.
- Columns: five.
- Positive labels: 113; negative labels: 127.
- SHA-256: `8cd27748b69d5cc03e9b2691d85c06d7a8d48814230aa82dbec451866214c4f2`.
- Machine-readable contract: `examples/schema.json`.
- Generator: `python -m examples.generate`.

Regeneration confirms byte identity and refuses to overwrite an existing different fixture. The example runner verifies the declared checksum, schema and row count before training.

## Partition and selection policy

The public baseline uses the explicit settings in `configs/synthetic_example.json`: seed 91357, 60% training, 20% validation and 20% test. Training and validation sizes use Python `round(n * ratio)`; test receives the remainder. In this fixture the sizes are 144/48/48.

The splitter uses two sequential scikit-learn stratified splits. Small or imbalanced samples may make the second split infeasible even when a different three-way allocation could exist. The function reports which stage failed; it does not implement a new constrained allocation algorithm.

The scaler and classifier fit only training rows. Validation and test are reporting-only. This example does not tune hyperparameters or thresholds, perform repeated-seed inference, or estimate uncertainty.

## Limitations

The simple target formula deliberately makes a small illustration, not a realistic benchmark. Single-seed metrics have no claimed statistical significance. Random row stratification does not demonstrate out-of-time, spatial, event or entity generalization. Synthetic fixture reproducibility does not establish reproduction of the private study.

The repository's current rights statement applies; no broad data or software license is invented here.
