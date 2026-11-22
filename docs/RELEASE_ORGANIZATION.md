# Academic repository organization

[GAT](https://github.com/PetarV-/GAT) provides model code, utilities, a minimal example, tested dependencies and a paper citation. [GraphSAGE](https://github.com/williamleif/GraphSAGE) separates small example data from full dataset downloads and documents input formats and execution.

This repository uses models/graph/losses/trainers/pipelines for the core algorithm, configs for executable examples, examples for independently generated data, tests for checks and docs for contracts. Referenced code and datasets were not copied.

The current release expands the previous auxiliary-only scope to include the original core implementation. Five processed research CSV files are now distributed; raw inputs and the experiment archive remain outside the repository.
