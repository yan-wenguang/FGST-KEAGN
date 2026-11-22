from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.io import load_json
from src.utils.metrics import stable_softmax


@dataclass
class PriorRecord:
    feature_name: str
    importance: float
    direction: str
    confidence: float
    evidence: str
    score: float


class LocalJSONKnowledgeProvider:
    def __init__(self, priors_path: str) -> None:
        self.priors_path = priors_path
        self.priors = load_json(priors_path, default={})


    def _fallback_prior(self, feature_name: str, feature_card: dict) -> PriorRecord:
        corr = float(feature_card.get("label_correlation", 0.0))
        importance = min(0.95, max(0.05, abs(corr)))
        direction = "positive" if corr >= 0 else "negative"
        confidence = 0.55
        evidence = "Fallback prior derived from feature-label correlation and generated feature card."
        score = importance * confidence
        return PriorRecord(feature_name, importance, direction, confidence, evidence, score)


    def build_global_prior(
        self,
        feature_names: list[str],
        feature_cards: dict[str, dict],
    ) -> tuple[np.ndarray, list[dict]]:
        records: list[PriorRecord] = []
        for feature_name in feature_names:
            if feature_name in self.priors:
                prior = self.priors[feature_name]
                importance = float(prior.get("importance", 0.5))
                confidence = float(prior.get("confidence", 0.5))
                direction = str(prior.get("direction", "conditional"))
                evidence = str(prior.get("evidence", "Loaded from local prior file."))
                records.append(
                    PriorRecord(
                        feature_name=feature_name,
                        importance=importance,
                        direction=direction,
                        confidence=confidence,
                        evidence=evidence,
                        score=importance * confidence,
                    )
                )
            else:
                records.append(self._fallback_prior(feature_name, feature_cards.get(feature_name, {})))

        weights = stable_softmax(np.asarray([record.score for record in records], dtype=np.float32))
        record_payload = [
            {
                "feature_name": record.feature_name,
                "importance": record.importance,
                "direction": record.direction,
                "confidence": record.confidence,
                "evidence": record.evidence,
                "score": record.score,
                "prior_weight": float(weight),
            }
            for record, weight in zip(records, weights, strict=True)
        ]
        return weights.astype(np.float32), record_payload

