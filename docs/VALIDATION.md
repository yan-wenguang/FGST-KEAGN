# Validation records

Before repository recreation, the same core source passed dependency checks, 44 tests, synthetic fixture generation, the auxiliary baseline and the full three-epoch core pipeline on Python 3.10 and 3.12 with PyTorch 2.11.0. Both environments selected feature_a and feature_c; synthetic stage-two test AUC was 0.4878. Those historical CI records belonged to the previous repository. New-repository CI results must be checked separately.

Five processed CSV inputs retain their source bytes. Checksums, dimensions, label counts and numeric completeness checks are recorded in data/processed/manifest.json. Real-data paper-result reproduction was not performed during this release.
