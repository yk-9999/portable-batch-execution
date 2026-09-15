import json
from pathlib import Path

import numpy as np

from portable_batch_execution.packs.ml import FakeEncoder, MLPack

FIXTURE = Path(__file__).parents[4] / "fixtures" / "public" / "ml" / "sentiment.json"


def dataset():
    return json.loads(FIXTURE.read_text())


def test_vectorizers_and_estimators():
    pack, data = MLPack(), dataset()
    tfidf = pack.execute("ml.tfidf", data["texts"])
    hashed = pack.execute("ml.hashing_vectorizer", data["texts"], n_features=32)
    assert tfidf.shape[0] == hashed.shape[0] == len(data["texts"])
    assert (
        len(pack.execute("ml.logistic_regression", tfidf, data["labels"]).classes_) == 2
    )
    assert len(pack.execute("ml.sgd_classifier", hashed, data["labels"]).classes_) == 2


def test_training_validation_and_inference():
    pack, data = MLPack(), dataset()
    split = pack.execute("ml.train_test", data["texts"], data["labels"], test_size=0.5)
    grouped = pack.execute(
        "ml.group_holdout",
        data["texts"],
        data["labels"],
        groups=data["groups"],
        test_size=0.5,
    )
    cv = pack.execute("ml.cross_validate", data["texts"], data["labels"], cv=2)
    calibrated = pack.execute("ml.calibrate", data["texts"], data["labels"], cv=2)
    inferred = pack.execute(
        "ml.batch_inference", data["texts"], model=split["model"], batch_size=2
    )
    assert split["metrics"]["accuracy"] >= 0
    assert {data["groups"][i] for i in grouped["train_indices"]}.isdisjoint(
        {data["groups"][i] for i in grouped["test_indices"]}
    )
    assert len(cv["scores"]) == 2
    assert calibrated.predict(data["texts"]).shape == (8,)
    assert len(inferred["predictions"]) == 8


def test_threshold_embedding_similarity_and_clustering():
    pack, data = MLPack(), dataset()
    sweep = pack.execute(
        "ml.threshold_sweep", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], thresholds=[0.25, 0.5]
    )
    embeddings = pack.execute("ml.embedding", data["texts"], encoder=FakeEncoder(8))
    similarity = pack.execute("ml.similarity", data["texts"], dimensions=8)
    clusters = pack.execute("ml.clustering", embeddings, n_clusters=2)
    assert sweep["best"]["f1"] == 1.0
    assert embeddings.shape == (8, 8)
    assert np.allclose(np.diag(similarity), 1.0)
    assert len(clusters["labels"]) == 8
