# Core implementation and execution contract

The core Python files are copied from the supplied `new_data` project. Algorithm formulas and training logic are preserved. The public run configuration uses the existing synthetic fixture, small model dimensions and three epochs; it is separate from private experiment settings.

## Stage one

Feature relations combine normalized mutual information, absolute correlation and classification synergy. The synergy implementation fits logistic regression on training rows and evaluates on those same training rows; the parameter named `synergy_cv_max_iter` sets iterations, not cross-validation. Attention-derived and embedding-derived scores form a soft feature mask. Redundancy-aware greedy selection uses validation AUC. Validation chooses checkpoints and feature subsets. Stage-one subset metrics may refit logistic regression on train plus validation rows.

## Stage two

Training-row means and standard deviations normalize inputs. Feature cards and local JSON priors feed the knowledge branch. Missing JSON entries fall back to feature-label correlations. Data/knowledge weights and context embeddings determine the gate and prediction. Validation AUC selects the checkpoint; outputs contain test probabilities and feature weights. There is no online LLM invocation in the current source.

## Input and outputs

Inputs are a numeric CSV, binary label column, explicit numeric context columns and a YAML configuration. Use positive ratios summing to one, sufficient examples of both classes and feature bounds compatible with the number of columns. Core modules use `src.*` imports from the repository root. Python 3.10+ is required because the original code uses `zip(..., strict=True)`.

The full runner passes stage one's selected-feature file into stage two. It writes models, relation matrices, histories, per-row predictions, feature weights and aggregate reports under the configured artifacts directories. These local generated outputs are ignored by Git.

## Evaluation limits

The core evaluator is retained from the original project: it maps positive labels to 1, substitutes nonfinite probabilities, uses fallback AUC for single-class truth and calls average precision `pr_auc`. This differs from the stricter auxiliary evaluator in `fgst_keagn_support`; the auxiliary evaluator's documented invalid-input and null-value contract does not apply to core metrics.

Random row stratification does not establish temporal, spatial or event-disjoint generalization. Running the core algorithm on synthetic data verifies software execution, not paper performance or scientific claims.
