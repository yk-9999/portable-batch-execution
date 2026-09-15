"""Deterministic, local implementations of the closed ML batch operations.

The pack deliberately accepts in-memory values.  Artifact loading and batch
orchestration are kernel concerns; keeping this boundary small makes every
operation suitable for an offline worker and for reproducible tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline


def _texts(values: Iterable[str]) -> list[str]:
    result = list(values)
    if not result or not all(isinstance(value, str) for value in result):
        raise ValueError("texts must be a non-empty sequence of strings")
    return result


def _labels(values: Sequence[Any] | None, expected_size: int) -> list[Any]:
    if values is None or len(values) != expected_size:
        raise ValueError("labels must have the same length as texts")
    if len(set(values)) < 2:
        raise ValueError("at least two label classes are required")
    return list(values)


def _vectorizer(kind: str, params: dict[str, Any]):
    allowed = {"max_features", "ngram_range", "lowercase", "stop_words"}
    options = {key: value for key, value in params.items() if key in allowed}
    if kind == "tfidf":
        return TfidfVectorizer(**options)
    if kind == "hashing":
        # A fixed dimensionality makes hashing stable across train and inference.
        return HashingVectorizer(n_features=params.get("n_features", 2**12), **options)
    raise ValueError("vectorizer must be 'tfidf' or 'hashing'")


def _classifier(kind: str, params: dict[str, Any]):
    if kind == "logistic_regression":
        return LogisticRegression(
            max_iter=params.get("max_iter", 500),
            C=params.get("C", 1.0),
            random_state=params.get("random_state", 0),
        )
    if kind == "sgd_classifier":
        return SGDClassifier(
            loss=params.get("loss", "log_loss"),
            max_iter=params.get("max_iter", 1000),
            tol=params.get("tol", 1e-3),
            random_state=params.get("random_state", 0),
        )
    raise ValueError("classifier must be 'logistic_regression' or 'sgd_classifier'")


@dataclass(frozen=True)
class FakeEncoder:
    """A local embedding boundary with stable, model-free vectors.

    It intentionally uses character buckets rather than downloading a model.
    The result is sufficient for exercising embedding, similarity, and
    clustering paths in offline environments.
    """

    dimensions: int = 16

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = _texts(texts)
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")
        matrix = np.zeros((len(values), self.dimensions), dtype=float)
        for row, text in enumerate(values):
            for char in text.lower():
                matrix[row, ord(char) % self.dimensions] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class MLPack:
    """Closed collection of reproducible ML operations backed by scikit-learn."""

    pack_id = "ml-batch"
    supported_operations = (
        "ml.tfidf",
        "ml.hashing_vectorizer",
        "ml.logistic_regression",
        "ml.sgd_classifier",
        "ml.train_test",
        "ml.group_holdout",
        "ml.cross_validate",
        "ml.calibrate",
        "ml.threshold_sweep",
        "ml.batch_inference",
        "ml.embedding",
        "ml.similarity",
        "ml.clustering",
    )

    def validate_params(self, operation: str, params: dict) -> dict:
        if operation not in self.supported_operations:
            raise ValueError("unsupported ml operation")
        if not isinstance(params, dict):
            raise TypeError("parameters must be a dictionary")
        return dict(params)

    def execute(self, operation: str, data=None, labels=None, **params):
        self.validate_params(operation, params)
        if operation == "ml.tfidf":
            return self.tfidf(data, **params)
        if operation == "ml.hashing_vectorizer":
            return self.hashing_vectorizer(data, **params)
        if operation == "ml.logistic_regression":
            return self.logistic_regression(data, labels, **params)
        if operation == "ml.sgd_classifier":
            return self.sgd_classifier(data, labels, **params)
        if operation == "ml.train_test":
            return self.train_test(data, labels, **params)
        if operation == "ml.group_holdout":
            return self.group_holdout(data, labels, **params)
        if operation == "ml.cross_validate":
            return self.cross_validate(data, labels, **params)
        if operation == "ml.calibrate":
            return self.calibrate(data, labels, **params)
        if operation == "ml.threshold_sweep":
            return self.threshold_sweep(labels, data, **params)
        if operation == "ml.batch_inference":
            return self.batch_inference(data, **params)
        if operation == "ml.embedding":
            return self.embedding(data, **params)
        if operation == "ml.similarity":
            return self.similarity(data, **params)
        return self.clustering(data, **params)

    def tfidf(self, texts: Iterable[str], **params):
        return _vectorizer("tfidf", params).fit_transform(_texts(texts))

    def hashing_vectorizer(self, texts: Iterable[str], **params):
        return _vectorizer("hashing", params).transform(_texts(texts))

    def logistic_regression(self, features, labels, **params):
        model = _classifier("logistic_regression", params)
        return model.fit(features, _labels(labels, features.shape[0]))

    def sgd_classifier(self, features, labels, **params):
        model = _classifier("sgd_classifier", params)
        return model.fit(features, _labels(labels, features.shape[0]))

    def _pipeline(self, params: dict[str, Any]) -> Pipeline:
        vectorizer = _vectorizer(params.get("vectorizer", "tfidf"), params)
        classifier = _classifier(
            params.get("classifier", "logistic_regression"), params
        )
        return Pipeline((("vectorizer", vectorizer), ("classifier", classifier)))

    @staticmethod
    def _metrics(truth, predicted) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(
                precision_score(truth, predicted, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(truth, predicted, average="weighted", zero_division=0)
            ),
            "f1": float(
                f1_score(truth, predicted, average="weighted", zero_division=0)
            ),
        }

    def train_test(self, texts, labels, **params):
        values = _texts(texts)
        targets = _labels(labels, len(values))
        indices = np.arange(len(values))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=params.get("test_size", 0.25),
            random_state=params.get("random_state", 0),
            stratify=targets if params.get("stratify", True) else None,
        )
        model = self._pipeline(params).fit(
            [values[i] for i in train_idx], [targets[i] for i in train_idx]
        )
        predicted = model.predict([values[i] for i in test_idx])
        truth = [targets[i] for i in test_idx]
        return {
            "model": model,
            "metrics": self._metrics(truth, predicted),
            "predictions": list(predicted),
            "test_indices": list(test_idx),
        }

    def group_holdout(self, texts, labels, *, groups, **params):
        values = _texts(texts)
        targets = _labels(labels, len(values))
        if len(groups) != len(values):
            raise ValueError("groups must have the same length as texts")
        splitter = GroupShuffleSplit(
            n_splits=16,
            test_size=params.get("test_size", 0.25),
            random_state=params.get("random_state", 0),
        )
        for train_idx, test_idx in splitter.split(values, targets, groups):
            if (
                len({targets[i] for i in train_idx}) > 1
                and len({targets[i] for i in test_idx}) > 1
            ):
                break
        else:
            raise ValueError("groups cannot produce a holdout with both label classes")
        model = self._pipeline(params).fit(
            [values[i] for i in train_idx], [targets[i] for i in train_idx]
        )
        predicted = model.predict([values[i] for i in test_idx])
        return {
            "model": model,
            "metrics": self._metrics([targets[i] for i in test_idx], predicted),
            "train_indices": list(train_idx),
            "test_indices": list(test_idx),
        }

    def cross_validate(self, texts, labels, **params):
        values = _texts(texts)
        targets = _labels(labels, len(values))
        folds = params.get("cv", 3)
        result = cross_validate(
            self._pipeline(params),
            values,
            targets,
            cv=folds,
            scoring=params.get("scoring", "accuracy"),
            return_train_score=False,
        )
        scores = [float(value) for value in result["test_score"]]
        return {"scores": scores, "mean_score": float(np.mean(scores)), "folds": folds}

    def calibrate(self, texts, labels, **params):
        values = _texts(texts)
        targets = _labels(labels, len(values))
        base = self._pipeline(params)
        model = CalibratedClassifierCV(
            base, method=params.get("method", "sigmoid"), cv=params.get("cv", 3)
        )
        return model.fit(values, targets)

    def threshold_sweep(self, labels, scores, **params):
        truth = _labels(labels, len(scores))
        probabilities = np.asarray(scores, dtype=float)
        if probabilities.ndim != 1 or not np.all(
            (0 <= probabilities) & (probabilities <= 1)
        ):
            raise ValueError("scores must be one-dimensional probabilities in [0, 1]")
        thresholds = params.get(
            "thresholds", [round(value, 2) for value in np.arange(0.0, 1.01, 0.05)]
        )
        rows = []
        for threshold in thresholds:
            predicted = probabilities >= threshold
            rows.append(
                {"threshold": float(threshold), **self._metrics(truth, predicted)}
            )
        best = max(
            rows, key=lambda row: (row[params.get("metric", "f1")], -row["threshold"])
        )
        return {"results": rows, "best": best}

    def batch_inference(
        self,
        texts,
        *,
        model,
        batch_size: int = 128,
        threshold: float | None = None,
        **_,
    ):
        values = _texts(texts)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions, probabilities = [], []
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            if threshold is not None and hasattr(model, "predict_proba"):
                proba = model.predict_proba(batch)
                positive = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                predictions.extend((positive >= threshold).tolist())
                probabilities.extend(positive.tolist())
            else:
                predictions.extend(model.predict(batch).tolist())
                if hasattr(model, "predict_proba"):
                    probabilities.extend(model.predict_proba(batch).tolist())
        return {"predictions": predictions, "probabilities": probabilities}

    def embedding(
        self, texts, *, encoder: Any | None = None, dimensions: int = 16, **_
    ):
        active_encoder = encoder or FakeEncoder(dimensions)
        if not hasattr(active_encoder, "encode"):
            raise ValueError("encoder must define encode(texts)")
        return np.asarray(active_encoder.encode(_texts(texts)), dtype=float)

    def similarity(self, items, *, other=None, **params):
        left = self._as_vectors(items, params)
        right = self._as_vectors(other, params) if other is not None else left
        return cosine_similarity(left, right)

    def clustering(self, items, *, n_clusters: int = 2, **params):
        vectors = self._as_vectors(items, params)
        if not 1 <= n_clusters <= len(vectors):
            raise ValueError("n_clusters must be between 1 and the number of samples")
        model = KMeans(
            n_clusters=n_clusters,
            random_state=params.get("random_state", 0),
            n_init=params.get("n_init", 10),
        ).fit(vectors)
        return {
            "labels": model.labels_.tolist(),
            "centers": model.cluster_centers_,
            "model": model,
        }

    def _as_vectors(self, items, params):
        values = list(items)
        if not values:
            raise ValueError("items must be non-empty")
        if all(isinstance(item, str) for item in values):
            return self.embedding(
                values,
                encoder=params.get("encoder"),
                dimensions=params.get("dimensions", 16),
            )
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("items must be a 2D numeric array or texts")
        return matrix
