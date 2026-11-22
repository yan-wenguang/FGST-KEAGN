from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.data.loader import frame_to_tensors, load_data_bundle
from src.data.splitter import stratified_split
from src.graph.adjacency import build_relation_matrices
from src.graph.node_features import build_node_features
from src.graph.redundancy import select_features_with_redundancy
from src.losses.stage1_loss import compute_stage1_loss
from src.models.fgat_fs import FGATFeatureSelector
from src.utils.io import ensure_dir, save_dataframe, save_json
from src.utils.metrics import compute_binary_metrics
from src.utils.seed import set_global_seed


def _build_standardization_stats(train_df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    means = train_df[feature_columns].mean(axis=0)
    stds = train_df[feature_columns].std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return means, stds


def _apply_standardization(
    df: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    scaled = df.copy()
    scaled = scaled.astype({column: float for column in feature_columns})
    scaled.loc[:, feature_columns] = (scaled[feature_columns] - means) / (stds + 1e-8)
    return scaled


def _evaluate_selector(model: FGATFeatureSelector, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        output = model(features)
        probabilities = torch.sigmoid(output["logits"]).cpu().numpy().reshape(-1)
        metrics = compute_binary_metrics(labels.cpu().numpy().reshape(-1), probabilities)
    return metrics


def _final_subset_metrics(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    label_column: str,
) -> dict[str, float]:
    x_train = train_val_df[selected_features].to_numpy(dtype=np.float32)
    y_train = train_val_df[label_column].to_numpy(dtype=int)
    x_test = test_df[selected_features].to_numpy(dtype=np.float32)
    y_test = test_df[label_column].to_numpy(dtype=int)
    if len(selected_features) == 1:
        raw_scores = x_test[:, 0]
        probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        model = LogisticRegression(max_iter=400, solver="lbfgs")
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
    return compute_binary_metrics(y_test, probabilities)


def train_stage1(config: dict) -> dict[str, object]:
    set_global_seed(int(config["seed"]))
    output_dir = ensure_dir(config["artifacts"]["stage1_dir"])
    data_bundle = load_data_bundle(
        config["data"]["csv_path"],
        config["data"]["label_column"],
        config["data"]["dynamic_context_columns"],
    )
    schema = data_bundle.schema
    splits = stratified_split(
        dataframe=data_bundle.dataframe,
        label_column=schema.label_column,
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
        test_ratio=float(config["data"]["test_ratio"]),
        seed=int(config["seed"]),
    )
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    normalize_inputs = bool(config.get("preprocessing", {}).get("normalize_inputs", True))
    if normalize_inputs:
        means, stds = _build_standardization_stats(train_df, schema.feature_columns)
        train_df = _apply_standardization(train_df, schema.feature_columns, means, stds)
        val_df = _apply_standardization(val_df, schema.feature_columns, means, stds)
        test_df = _apply_standardization(test_df, schema.feature_columns, means, stds)

    relation_matrices_np = build_relation_matrices(
        train_df,
        schema.feature_columns,
        schema.label_column,
        bins=int(config["stage1"]["mi_bins"]),
        max_iter=int(config["stage1"]["synergy_cv_max_iter"]),
    )
    node_features_np = build_node_features(
        train_df,
        schema.feature_columns,
        schema.label_column,
        bins=int(config["stage1"]["mi_bins"]),
    )

    train_tensors = frame_to_tensors(train_df, schema.feature_columns, schema.label_column)
    val_tensors = frame_to_tensors(val_df, schema.feature_columns, schema.label_column)
    test_tensors = frame_to_tensors(test_df, schema.feature_columns, schema.label_column)

    relation_tensors = {key: torch.tensor(value, dtype=torch.float32) for key, value in relation_matrices_np.items()}
    model = FGATFeatureSelector(
        node_features=torch.tensor(node_features_np, dtype=torch.float32),
        relation_matrices=relation_tensors,
        hidden_dim=int(config["stage1"]["hidden_dim"]),
        num_heads=int(config["stage1"]["num_heads"]),
        dropout=float(config["stage1"]["dropout"]),
        alpha_balance=float(config["stage1"]["alpha_balance"]),
        temperature=float(config["stage1"]["temperature"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["stage1"]["learning_rate"]),
        weight_decay=float(config["stage1"]["weight_decay"]),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -np.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(config["stage1"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        output = model(train_tensors["x"])
        loss, loss_metrics = compute_stage1_loss(
            logits=output["logits"],
            labels=train_tensors["y"],
            mask=output["mask"],
            sparse_weight=float(config["stage1"]["sparse_weight"]),
        )
        loss.backward()
        optimizer.step()

        train_metrics = compute_binary_metrics(
            train_tensors["y"].cpu().numpy().reshape(-1),
            torch.sigmoid(output["logits"]).detach().cpu().numpy().reshape(-1),
        )
        val_metrics = _evaluate_selector(model, val_tensors["x"], val_tensors["y"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_metrics["total_loss"],
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
            }
        )
        if val_metrics["auc"] >= best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_output = model(train_tensors["x"])
    scores = final_output["scores"].cpu().numpy()
    mask = final_output["mask"].cpu().numpy()
    redundancy_matrix = 0.5 * (relation_matrices_np["mi"] + relation_matrices_np["corr"])
    selection = select_features_with_redundancy(
        scores=scores,
        redundancy_matrix=redundancy_matrix,
        feature_names=schema.feature_columns,
        x_train=train_df[schema.feature_columns].to_numpy(dtype=np.float32),
        y_train=train_df[schema.label_column].to_numpy(dtype=int),
        x_val=val_df[schema.feature_columns].to_numpy(dtype=np.float32),
        y_val=val_df[schema.label_column].to_numpy(dtype=int),
        min_features=int(config["stage1"]["min_features"]),
        max_features=int(config["stage1"]["max_features"]),
        penalty=float(config["stage1"]["redundancy_penalty"]),
    )
    selected_features = selection["selected_names"]
    selected_set = set(selected_features)

    feature_frame = pd.DataFrame(
        {
            "feature_name": schema.feature_columns,
            "score": scores,
            "soft_mask": mask,
            "selected": [int(name in selected_set) for name in schema.feature_columns],
        }
    ).sort_values(by="score", ascending=False)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    subset_metrics = _final_subset_metrics(train_val_df, test_df, selected_features, schema.label_column)
    selector_test_metrics = _evaluate_selector(model, test_tensors["x"], test_tensors["y"])

    np.savez(
        Path(output_dir) / "relation_matrices.npz",
        mi=relation_matrices_np["mi"],
        corr=relation_matrices_np["corr"],
        cls=relation_matrices_np["cls"],
        redundancy=redundancy_matrix,
    )
    torch.save(model.state_dict(), Path(output_dir) / "fgat_fs_model.pt")
    save_dataframe(Path(output_dir) / "feature_scores.csv", feature_frame)
    save_dataframe(Path(output_dir) / "training_history.csv", pd.DataFrame(history))
    save_json(
        Path(output_dir) / "selected_features.json",
        {
            "selected_features": selected_features,
            "best_val_auc": selection["best_val_auc"],
            "history": selection["history"],
        },
    )
    save_json(
        Path(output_dir) / "stage1_metrics.json",
        {
            "selector_test_metrics": selector_test_metrics,
            "subset_test_metrics": subset_metrics,
            "relation_weights": final_output["relation_weights"].cpu().numpy().round(6).tolist(),
        },
    )

    return {
        "output_dir": str(output_dir),
        "selected_features": selected_features,
        "selector_test_metrics": selector_test_metrics,
        "subset_test_metrics": subset_metrics,
    }
