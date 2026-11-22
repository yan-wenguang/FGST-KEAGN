from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_data_bundle
from src.data.splitter import stratified_split
from src.knowledge.feature_cards import generate_feature_cards
from src.knowledge.providers import LocalJSONKnowledgeProvider
from src.losses.stage2_loss import compute_stage2_loss
from src.models.llm_keagn import LLMKEAGN
from src.utils.io import ensure_dir, load_json, save_dataframe, save_json
from src.utils.metrics import compute_binary_metrics
from src.utils.seed import set_global_seed


def _build_standardization_stats(train_df: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    means = train_df[columns].mean(axis=0)
    stds = train_df[columns].std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return means, stds


def _apply_standardization(
    df: pd.DataFrame,
    columns: list[str],
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    scaled = df.copy()
    scaled = scaled.astype({column: float for column in columns})
    scaled.loc[:, columns] = (scaled[columns] - means) / (stds + 1e-8)
    return scaled


def _make_loader(
    dataframe: pd.DataFrame,
    selected_features: list[str],
    context_columns: list[str],
    label_column: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_selected = torch.tensor(dataframe[selected_features].to_numpy(dtype=np.float32), dtype=torch.float32)
    x_context = torch.tensor(dataframe[context_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    labels = torch.tensor(dataframe[label_column].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
    dataset = TensorDataset(x_selected, x_context, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_with_model(
    model: LLMKEAGN,
    dataframe: pd.DataFrame,
    selected_features: list[str],
    context_columns: list[str],
    label_column: str,
    global_prior: torch.Tensor,
) -> dict[str, object]:
    model.eval()
    x_selected = torch.tensor(dataframe[selected_features].to_numpy(dtype=np.float32), dtype=torch.float32)
    x_context = torch.tensor(dataframe[context_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    labels = dataframe[label_column].to_numpy(dtype=int)
    with torch.no_grad():
        output = model(x_selected, x_context, global_prior)
        probabilities = torch.sigmoid(output["logits"]).cpu().numpy().reshape(-1)
    metrics = compute_binary_metrics(labels, probabilities)
    return {"output": output, "probabilities": probabilities, "metrics": metrics}


def train_stage2(config: dict) -> dict[str, object]:
    set_global_seed(int(config["seed"]))
    output_dir = ensure_dir(config["artifacts"]["stage2_dir"])
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

    ablation_config = config.get("ablation", {})
    use_feature_selection = bool(ablation_config.get("use_feature_selection", True))
    use_knowledge_enhancement = bool(ablation_config.get("use_knowledge_enhancement", True))
    use_adaptive_gate = bool(ablation_config.get("use_adaptive_gate", True))

    if use_feature_selection:
        selected_payload = load_json(config["knowledge"]["selected_features_path"], default={})
        selected_features = selected_payload.get("selected_features", [])
        if not selected_features:
            raise ValueError("Selected features are missing. Run stage 1 first or provide selected_features_path.")
    else:
        selected_features = list(schema.feature_columns)

    normalize_inputs = bool(config.get("preprocessing", {}).get("normalize_inputs", True))
    scaling_columns = sorted(set(selected_features + schema.dynamic_context_columns))
    if normalize_inputs:
        means, stds = _build_standardization_stats(train_df, scaling_columns)
        train_df = _apply_standardization(train_df, scaling_columns, means, stds)
        val_df = _apply_standardization(val_df, scaling_columns, means, stds)
        test_df = _apply_standardization(test_df, scaling_columns, means, stds)
        full_df = _apply_standardization(data_bundle.dataframe, scaling_columns, means, stds)
    else:
        full_df = data_bundle.dataframe

    feature_metadata = load_json(config["knowledge"]["feature_metadata_path"], default={})
    feature_cards = generate_feature_cards(
        dataframe=train_df,
        feature_names=selected_features,
        label_column=schema.label_column,
        feature_metadata=feature_metadata,
    )
    save_json(config["knowledge"]["feature_cards_path"], feature_cards)

    if use_knowledge_enhancement:
        provider = LocalJSONKnowledgeProvider(config["knowledge"]["priors_path"])
        global_prior_np, prior_records = provider.build_global_prior(selected_features, feature_cards)
    else:
        global_prior_np = np.full(len(selected_features), 1.0 / max(len(selected_features), 1), dtype=np.float32)
        prior_records = [
            {
                "feature_name": feature_name,
                "importance": 1.0 / max(len(selected_features), 1),
                "direction": "neutral",
                "confidence": 1.0,
                "evidence": "Knowledge enhancement ablated: using uniform prior.",
                "score": 1.0 / max(len(selected_features), 1),
                "prior_weight": 1.0 / max(len(selected_features), 1),
            }
            for feature_name in selected_features
        ]
    global_prior = torch.tensor(global_prior_np, dtype=torch.float32)

    model = LLMKEAGN(
        feature_dim=len(selected_features),
        context_input_dim=len(schema.dynamic_context_columns),
        hidden_dim=int(config["stage2"]["hidden_dim"]),
        context_dim=int(config["stage2"]["context_dim"]),
        dropout=0.10,
        use_knowledge_branch=use_knowledge_enhancement,
        use_adaptive_gate=use_adaptive_gate,
        weight_mode=str(config["stage2"].get("weight_mode", "softmax")),
        residual_raw_weight=float(config["stage2"].get("residual_raw_weight", 0.0)),
        raw_branch_weight=float(config["stage2"].get("raw_branch_weight", 0.0)),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["stage2"]["learning_rate"]),
        weight_decay=float(config["stage2"]["weight_decay"]),
    )

    train_loader = _make_loader(
        train_df,
        selected_features,
        schema.dynamic_context_columns,
        schema.label_column,
        batch_size=int(config["stage2"]["batch_size"]),
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -np.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(config["stage2"]["epochs"]) + 1):
        model.train()
        epoch_losses = []
        for x_selected, x_context, labels in train_loader:
            optimizer.zero_grad()
            output = model(x_selected, x_context, global_prior)
            loss, loss_metrics = compute_stage2_loss(
                logits=output["logits"],
                labels=labels,
                w_data=output["w_data"],
                w_knowledge=output["w_knowledge"],
                gate=output["gate"],
                context_embedding=output["context_embedding"],
                align_weight=float(config["stage2"]["align_weight"]) if use_knowledge_enhancement else 0.0,
                context_weight=float(config["stage2"]["context_weight"]) if use_adaptive_gate else 0.0,
                context_gamma=float(config["stage2"]["context_kernel_gamma"]),
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_metrics["total_loss"])

        train_result = _predict_with_model(
            model,
            train_df,
            selected_features,
            schema.dynamic_context_columns,
            schema.label_column,
            global_prior,
        )
        val_result = _predict_with_model(
            model,
            val_df,
            selected_features,
            schema.dynamic_context_columns,
            schema.label_column,
            global_prior,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "train_auc": train_result["metrics"]["auc"],
                "val_auc": val_result["metrics"]["auc"],
            }
        )
        if val_result["metrics"]["auc"] >= best_val_auc:
            best_val_auc = val_result["metrics"]["auc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    train_result = _predict_with_model(
        model,
        train_df,
        selected_features,
        schema.dynamic_context_columns,
        schema.label_column,
        global_prior,
    )
    val_result = _predict_with_model(
        model,
        val_df,
        selected_features,
        schema.dynamic_context_columns,
        schema.label_column,
        global_prior,
    )
    test_result = _predict_with_model(
        model,
        test_df,
        selected_features,
        schema.dynamic_context_columns,
        schema.label_column,
        global_prior,
    )
    full_result = _predict_with_model(
        model,
        full_df,
        selected_features,
        schema.dynamic_context_columns,
        schema.label_column,
        global_prior,
    )

    with torch.no_grad():
        fused_weights = full_result["output"]["fused_weights"].cpu().numpy()
        global_feature_weights = fused_weights.mean(axis=0)

    global_feature_frame = pd.DataFrame(
        {"feature_name": selected_features, "global_weight": global_feature_weights}
    ).sort_values(by="global_weight", ascending=False)
    prediction_frame = test_df[selected_features + schema.dynamic_context_columns + [schema.label_column]].copy()
    prediction_frame["predicted_probability"] = test_result["probabilities"]
    prediction_frame["predicted_label"] = (prediction_frame["predicted_probability"] >= 0.5).astype(int)
    val_prediction_frame = val_df[selected_features + schema.dynamic_context_columns + [schema.label_column]].copy()
    val_prediction_frame["predicted_probability"] = val_result["probabilities"]
    val_prediction_frame["predicted_label"] = (val_prediction_frame["predicted_probability"] >= 0.5).astype(int)

    sample_weights_frame = pd.DataFrame(
        test_result["output"]["fused_weights"].cpu().numpy(),
        columns=[f"weight::{name}" for name in selected_features],
    )
    sample_weights_frame.insert(0, "index", np.arange(len(sample_weights_frame)))

    save_dataframe(Path(output_dir) / "training_history.csv", pd.DataFrame(history))
    save_dataframe(Path(output_dir) / "val_predictions.csv", val_prediction_frame.reset_index(drop=True))
    save_dataframe(Path(output_dir) / "predictions.csv", prediction_frame.reset_index(drop=True))
    save_dataframe(Path(output_dir) / "sample_feature_weights.csv", sample_weights_frame)
    save_dataframe(Path(output_dir) / "global_feature_weights.csv", global_feature_frame)
    save_json(Path(output_dir) / "knowledge_prior_records.json", prior_records)
    save_json(
        Path(output_dir) / "stage2_metrics.json",
        {
            "train": train_result["metrics"],
            "val": val_result["metrics"],
            "test": test_result["metrics"],
        },
    )
    torch.save(model.state_dict(), Path(output_dir) / "llm_keagn_model.pt")

    return {
        "output_dir": str(output_dir),
        "test_metrics": test_result["metrics"],
        "global_feature_weights": global_feature_frame.to_dict(orient="records"),
        "selected_features": selected_features,
        "ablation": {
            "use_feature_selection": use_feature_selection,
            "use_knowledge_enhancement": use_knowledge_enhancement,
            "use_adaptive_gate": use_adaptive_gate,
        },
    }
