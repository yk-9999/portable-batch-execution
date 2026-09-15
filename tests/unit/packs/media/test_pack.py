from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

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
