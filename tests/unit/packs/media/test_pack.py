from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from portable_batch_execution.packs.media import MediaPack

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and FFprobe are required for media smoke tests",
)


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    """Build a short synthetic stereo WAV without committing binary fixtures."""
    output = tmp_path / "stereo.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:sample_rate=48000:duration=2",
            "-filter_complex",
            "[0:a][1:a]join=inputs=2:channel_layout=stereo",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
        shell=False,
    )
    return output


def test_resample_mono_segment_and_metadata_smoke(
    stereo_wav: Path, tmp_path: Path
) -> None:
    pack = MediaPack()
    resampled = pack.resample(stereo_wav, tmp_path / "resampled.wav", 16_000)
    mono = pack.mono(resampled, tmp_path / "mono.wav")
    metadata = pack.metadata(mono)
    pieces = pack.segment(mono, tmp_path / "segments", 0.75)

    assert metadata["audio"]["sample_rate"] == 16_000
    assert metadata["audio"]["channels"] == 1
    assert metadata["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert len(pieces) == 3
    assert all(piece.is_file() for piece in pieces)


def test_decode_extract_merge_and_asr_overlap_removal(
    stereo_wav: Path, tmp_path: Path
) -> None:
    pack = MediaPack()
    decoded = pack.decode(stereo_wav, tmp_path / "decoded.wav")
    extracted = pack.extract_audio(stereo_wav, tmp_path / "extracted.wav")
    merged = pack.merge([decoded, extracted], tmp_path / "merged.wav")

    assert pack.metadata(merged)["duration_seconds"] == pytest.approx(4.0, abs=0.1)
    assert pack.asr_merge(
        [
            {"segments": [{"start": 0.0, "end": 1.0, "text": "hello there"}]},
            {"segments": [{"start": 0.8, "end": 2.0, "text": "there friend"}]},
        ]
    ) == [
        {"start": 0.0, "end": 1.0, "text": "hello there"},
        {"start": 1.0, "end": 2.0, "text": "friend"},
    ]


def test_params_are_closed() -> None:
    pack = MediaPack()
    with pytest.raises(ValueError, match="invalid media operation parameters"):
        pack.validate_params(
            "media.mono", {"input": "a.wav", "output": "b.wav", "cmd": "x"}
        )
    with pytest.raises(ValueError, match="unsupported media operation"):
        pack.validate_params("media.other", {})


def test_asr_normalize_flac_params_must_be_empty() -> None:
    pack = MediaPack()
    assert pack.validate_params("media.asr_normalize_flac", {}) == {}
    with pytest.raises(ValueError, match="invalid media operation parameters"):
        pack.validate_params("media.asr_normalize_flac", {"input": "a.bin"})


def test_asr_normalize_flac_uses_fixed_ffmpeg_argv(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"\x00")
    destination = tmp_path / "output.flac"
    pack = MediaPack(ffmpeg="ffmpeg-bin", ffprobe="ffprobe-bin")
    captured: list[list[str]] = []

    def fake_run(args: list[str]) -> None:
        captured.append(args)
        destination.write_bytes(b"flac")

    with patch.object(pack, "_run", side_effect=fake_run):
        pack.asr_normalize_flac(source, destination)

    assert captured == [
        [
            "ffmpeg-bin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            str(destination),
        ]
    ]


def test_asr_normalize_flac_execute_validates_worker_context(tmp_path: Path) -> None:
    pack = MediaPack()
    source = tmp_path / "input.bin"
    source.write_bytes(b"\x00")
    destination = tmp_path / "output.flac"
    with patch.object(pack, "asr_normalize_flac", return_value=destination) as called:
        pack.execute(
            {"operation": "media.asr_normalize_flac"},
            None,
            {},
            {"input": str(source), "output": str(destination)},
        )
    called.assert_called_once_with(source, destination)
    with pytest.raises(ValueError, match="invalid media worker context"):
        pack.execute(
            {"operation": "media.asr_normalize_flac"},
            None,
            {},
            {"input": str(source)},
        )
