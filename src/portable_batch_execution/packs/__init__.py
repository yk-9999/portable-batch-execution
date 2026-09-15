"""Closed, small public pack operations used by synthetic workloads."""

from __future__ import annotations

import statistics


def tabular(operation: str, rows: list[dict], **p):
    import polars as pl

    df = pl.DataFrame(rows)
    if operation == "tabular.normalize":
        return df.fill_null(0).to_dicts()
    if operation == "tabular.sort":
        return df.sort(p["by"]).to_dicts()
    if operation == "tabular.dedup":
        return df.unique(subset=p.get("by")).to_dicts()
    if operation == "tabular.rolling":
        return df.with_columns(
            pl.col(p["column"]).rolling_mean(p["window"]).alias("rolling")
        ).to_dicts()
    if operation == "tabular.statistics":
        return {"count": df.height, "mean": statistics.mean(df[p["column"]].to_list())}
    raise ValueError("unsupported tabular parameters")


def acquisition(operation: str, pages: list[dict], **p):
    if operation not in {
        "acquisition.rest",
        "acquisition.html",
        "acquisition.incremental",
    }:
        raise ValueError("unsupported acquisition")
    return [x for x in pages if x.get("id", 0) >= p.get("since", 0)]


def replay_eval(operation: str, values: list[float], **p):
    if not operation.startswith("replay_eval."):
        raise ValueError("unsupported replay")
    return {"count": len(values), "mean": sum(values) / len(values) if values else 0.0}


def ml(operation: str, texts: list[str], labels: list[int] | None = None):
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

    if operation == "ml.tfidf":
        return TfidfVectorizer().fit_transform(texts).shape
    if operation == "ml.hashing_vectorizer":
        return HashingVectorizer(n_features=16).transform(texts).shape
    raise ValueError("unsupported ml parameters")


def media(operation: str, segments: list[dict], **p):
    if operation == "media.metadata":
        return {
            "segments": len(segments),
            "duration": sum(x["end"] - x["start"] for x in segments),
        }
    if operation == "media.overlap_remove":
        out = []
        for s in sorted(segments, key=lambda x: x["start"]):
            if not out or s["start"] >= out[-1]["end"]:
                out.append(s)
        return out
    raise ValueError("unsupported media parameters")
