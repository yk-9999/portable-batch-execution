from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from portable_batch_execution.packs.tabular import TabularPack, rolling_halo


def test_strict_models_and_closed_operation():
    pack = TabularPack()
    assert pack.validate_params("tabular.sort", {"by": [{"column": "a"}]}) == {
        "by": [{"column": "a", "descending": False, "nulls_last": False}]
    }
    with pytest.raises(ValidationError):
        pack.validate_params(
            "tabular.sort", {"by": [{"column": "a"}], "sql": "select 1"}
        )
    with pytest.raises(ValueError):
        pack.validate_params("tabular.sql", {})


def test_normalize_cast_sort_dedup_and_statistics():
    pack = TabularPack()
    rows = [
        {"group": " B ", "value": "2"},
        {"group": "a", "value": "2"},
        {"group": "a", "value": "1"},
    ]
    normalized = pack.run(
        "tabular.normalize", rows, {"columns": ["group"], "lowercase": True}
    )
    cast = pack.run("tabular.cast", normalized, {"columns": {"value": "integer"}})
    sorted_rows = pack.run("tabular.sort", cast, {"by": [{"column": "value"}]})
    deduped = pack.run("tabular.dedup", sorted_rows, {"subset": ["value"]})
    stats = pack.run(
        "tabular.statistics",
        deduped,
        {"columns": ["value"], "aggregations": ["count", "sum"]},
    )
    assert deduped["value"].to_list() == [1, 2]
    assert stats.row(0) == (2, 3)


def test_join_pit_boundary_window_rolling_and_halo():
    pack = TabularPack()
    left = [
        {"id": "a", "at": 10, "v": 1},
        {"id": "a", "at": 20, "v": 2},
        {"id": "b", "at": 10, "v": 3},
    ]
    right = [
        {"id": "a", "seen": 10, "label": "exact"},
        {"id": "a", "seen": 15, "label": "past"},
        {"id": "b", "seen": 9, "label": "early"},
    ]
    joined = pack.run("tabular.join", left, {"on": ["id"], "how": "left"}, right=right)
    pit = pack.run(
        "tabular.pit_join",
        left,
        {"on": ["id"], "left_time": "at", "right_time": "seen"},
        right=right,
    )
    windowed = pack.run(
        "tabular.window",
        left,
        {
            "partition_by": ["id"],
            "order_by": [{"column": "at"}],
            "function": "row_number",
            "output_column": "n",
        },
    )
    rolled = pack.run(
        "tabular.rolling",
        left,
        {
            "column": "v",
            "window_size": 2,
            "partition_by": ["id"],
            "order_by": [{"column": "at"}],
            "output_column": "avg",
        },
    )
    assert joined.height == 5
    assert (
        pit.filter(pl.col("at") == 10).filter(pl.col("id") == "a")["label"].item()
        == "exact"
    )
    assert windowed.filter(pl.col("id") == "a")["n"].to_list() == [1, 2]
    assert rolled.filter(pl.col("id") == "a")["avg"].to_list() == [1.0, 1.5]
    assert rolling_halo(2) == 1


@pytest.mark.parametrize("extension", ["csv", "json", "jsonl", "parquet"])
def test_all_supported_file_formats(tmp_path: Path, extension: str):
    pack = TabularPack()
    source = tmp_path / f"source.{extension}"
    target = tmp_path / f"target.{extension}"
    pl.DataFrame({"a": [1, 2]}).write_csv(source) if extension == "csv" else (
        pl.DataFrame({"a": [1, 2]}).write_json(source)
        if extension == "json"
        else (
            pl.DataFrame({"a": [1, 2]}).write_ndjson(source)
            if extension == "jsonl"
            else pl.DataFrame({"a": [1, 2]}).write_parquet(source)
        )
    )
    result = pack.run(
        "tabular.format_migration",
        source,
        {"output_format": extension},
        destination=target,
    )
    assert result == target
    assert pack.run("tabular.sort", target, {"by": [{"column": "a"}]}).height == 2
