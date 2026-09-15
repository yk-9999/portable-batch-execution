"""A deliberately small, shell-free FFmpeg media pack.

The pack only builds argv lists from validated values.  It never accepts a
filter expression, codec name, or command string from a job parameter.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar


class MediaPack:
    """Closed media transformations suitable for reproducible batch shards."""

    pack_id = "media-batch"
    supported_operations = (
        "media.decode",
        "media.extract_audio",
        "media.resample",
        "media.mono",
        "media.segment",
        "media.merge",
        "media.metadata",
        "media.asr_merge",
        "media.overlap_remove",
    )

    _AUDIO_SUFFIXES: ClassVar[set[str]] = {
        ".wav",
        ".wave",
        ".mp3",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
    }

    def __init__(self, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        # These are constructor-owned trusted executable names, rather than job
        # parameters.  This makes the command surface independent of a manifest.
        if not ffmpeg or not ffprobe or "\x00" in ffmpeg + ffprobe:
            raise ValueError("valid FFmpeg executable names are required")
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    @staticmethod
    def _path(value: str | Path, *, output: bool = False) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError("media paths must be strings or Path objects")
        raw = str(value)
        if not raw or "\x00" in raw:
            raise ValueError("valid local media path required")
        path = Path(raw)
        if path.name.startswith("-"):
            raise ValueError("media filenames may not begin with '-'")
        if not output and not path.is_file():
            raise FileNotFoundError(path)
        if output:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _seconds(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        number = float(value)
        if number <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return number

    @staticmethod
    def _sample_rate(value: Any) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 8_000 <= value <= 192_000
        ):
            raise ValueError("sample_rate must be an integer between 8000 and 192000")
        return value

    def _run(self, args: list[str]) -> None:
        subprocess.run(args, check=True, shell=False, capture_output=True, text=True)

    def _ffmpeg_args(self, *args: str) -> list[str]:
        return [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *args]

    def decode(self, source: str | Path, destination: str | Path) -> Path:
        """Decode the first audio stream to deterministic PCM WAV."""
        src, dst = self._path(source), self._path(destination, output=True)
        self._run(
            self._ffmpeg_args(
                "-y",
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "pcm_s16le",
                str(dst),
            )
        )
        return dst

    def extract_audio(self, source: str | Path, destination: str | Path) -> Path:
        """Extract and decode the first audio stream to PCM WAV."""
        return self.decode(source, destination)

    def resample(
        self, source: str | Path, destination: str | Path, sample_rate: int
    ) -> Path:
        src, dst = self._path(source), self._path(destination, output=True)
        rate = self._sample_rate(sample_rate)
        self._run(
            self._ffmpeg_args(
                "-y",
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-vn",
                "-ar",
                str(rate),
                "-c:a",
                "pcm_s16le",
                str(dst),
            )
        )
        return dst

    def mono(self, source: str | Path, destination: str | Path) -> Path:
        src, dst = self._path(source), self._path(destination, output=True)
        self._run(
            self._ffmpeg_args(
                "-y",
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(dst),
            )
        )
        return dst

    def segment(
        self, source: str | Path, destination_dir: str | Path, duration_seconds: float
    ) -> list[Path]:
        src = self._path(source)
        duration = self._seconds(duration_seconds, "duration_seconds")
        directory = self._path(destination_dir, output=True)
        directory.mkdir(parents=True, exist_ok=True)
        pattern = directory / "segment-%04d.wav"
        self._run(
            self._ffmpeg_args(
                "-y",
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "pcm_s16le",
                "-f",
                "segment",
                "-segment_time",
                f"{duration:g}",
                "-reset_timestamps",
                "1",
                str(pattern),
            )
        )
        return sorted(directory.glob("segment-*.wav"))

    def merge(self, sources: Sequence[str | Path], destination: str | Path) -> Path:
        if not sources:
            raise ValueError("at least one source is required")
        paths = [self._path(item) for item in sources]
        if any(path.suffix.lower() not in self._AUDIO_SUFFIXES for path in paths):
            raise ValueError("merge accepts audio files only")
        dst = self._path(destination, output=True)
        # concat receives a fixed number of labeled inputs and fixed filter text;
        # neither comes from a job supplied command or filter expression.
        inputs = [part for path in paths for part in ("-i", str(path))]
        labels = "".join(f"[{index}:a]" for index in range(len(paths)))
        graph = f"{labels}concat=n={len(paths)}:v=0:a=1[outa]"
        self._run(
            self._ffmpeg_args(
                "-y",
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                "[outa]",
                "-c:a",
                "pcm_s16le",
                str(dst),
            )
        )
        return dst

    def metadata(self, source: str | Path) -> dict[str, Any]:
        src = self._path(source)
        completed = subprocess.run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(src),
            ],
            check=True,
            shell=False,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        audio = next(
            (
                stream
                for stream in payload.get("streams", [])
                if stream.get("codec_type") == "audio"
            ),
            {},
        )
        format_info = payload.get("format", {})
        duration = format_info.get("duration") or audio.get("duration")
        return {
            "path": str(src),
            "format": format_info.get("format_name"),
            "duration_seconds": float(duration) if duration is not None else None,
            "size_bytes": int(format_info["size"])
            if format_info.get("size") is not None
            else src.stat().st_size,
            "audio": {
                "codec": audio.get("codec_name"),
                "sample_rate": int(audio["sample_rate"])
                if audio.get("sample_rate")
                else None,
                "channels": audio.get("channels"),
            },
        }

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\S+", text)

    @classmethod
    def _overlap_words(cls, previous: str, current: str) -> int:
        left, right = cls._tokens(previous), cls._tokens(current)
        limit = min(len(left), len(right))
        for size in range(limit, 0, -1):
            if [x.casefold().strip(".,!?;:") for x in left[-size:]] == [
                x.casefold().strip(".,!?;:") for x in right[:size]
            ]:
                return size
        return 0

    @classmethod
    def overlap_remove(
        cls, segments: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicated ASR words and trim temporal overlap when possible."""
        ordered = sorted(
            (dict(item) for item in segments),
            key=lambda item: (item.get("start", 0), item.get("end", 0)),
        )
        out: list[dict[str, Any]] = []
        for item in ordered:
            if (
                "text" in item
                and out
                and isinstance(item["text"], str)
                and isinstance(out[-1].get("text"), str)
            ):
                words = cls._tokens(item["text"])
                overlap = cls._overlap_words(out[-1]["text"], item["text"])
                item["text"] = " ".join(words[overlap:])
                if not item["text"]:
                    continue
            if (
                out
                and isinstance(item.get("start"), (int, float))
                and isinstance(out[-1].get("end"), (int, float))
                and item["start"] < out[-1]["end"]
            ):
                item["start"] = out[-1]["end"]
                if (
                    isinstance(item.get("end"), (int, float))
                    and item["end"] <= item["start"]
                ):
                    continue
            out.append(item)
        return out

    @classmethod
    def asr_merge(
        cls, transcripts: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Flatten shard ASR results, order them, and remove boundary duplicates."""
        segments: list[dict[str, Any]] = []
        for transcript in transcripts:
            values = transcript.get("segments", [transcript])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError("ASR transcript segments must be a sequence")
            for value in values:
                if not isinstance(value, Mapping):
                    raise TypeError("ASR segment must be an object")
                if not isinstance(value.get("text"), str):
                    raise TypeError("ASR segment text is required")
                segments.append(dict(value))
        return cls.overlap_remove(segments)

    # Friendly aliases for callers that use verb-first names.
    remove_overlaps = overlap_remove
    merge_asr = asr_merge

    def validate_params(self, operation: str, params: dict) -> dict:
        if operation not in self.supported_operations or not isinstance(params, dict):
            raise ValueError("unsupported media operation")
        allowed = {
            "media.decode": {"input", "output"},
            "media.extract_audio": {"input", "output"},
            "media.resample": {"input", "output", "sample_rate"},
            "media.mono": {"input", "output"},
            "media.segment": {"input", "output_dir", "duration_seconds"},
            "media.merge": {"inputs", "output"},
            "media.metadata": {"input"},
            "media.asr_merge": {"transcripts"},
            "media.overlap_remove": {"segments"},
        }[operation]
        if set(params) != allowed:
            raise ValueError("invalid media operation parameters")
        if operation == "media.resample":
            self._sample_rate(params["sample_rate"])
        if operation == "media.segment":
            self._seconds(params["duration_seconds"], "duration_seconds")
        return dict(params)

    def execute(self, job: Any, shard: Any, params: dict, context: Any) -> Any:
        operation = getattr(job, "operation", None) or (
            job.get("operation") if isinstance(job, Mapping) else None
        )
        checked = self.validate_params(operation, params)
        if operation == "media.decode":
            return self.decode(checked["input"], checked["output"])
        if operation == "media.extract_audio":
            return self.extract_audio(checked["input"], checked["output"])
        if operation == "media.resample":
            return self.resample(
                checked["input"], checked["output"], checked["sample_rate"]
            )
        if operation == "media.mono":
            return self.mono(checked["input"], checked["output"])
        if operation == "media.segment":
            return self.segment(
                checked["input"], checked["output_dir"], checked["duration_seconds"]
            )
        if operation == "media.merge":
            return self.merge(checked["inputs"], checked["output"])
        if operation == "media.metadata":
            return self.metadata(checked["input"])
        if operation == "media.asr_merge":
            return self.asr_merge(checked["transcripts"])
        return self.overlap_remove(checked["segments"])

    def finalize(self, job: Any, canonical_attempts: Any, context: Any) -> Any:
        """Media finalization is intentionally delegated to the explicit merge ops."""
        return canonical_attempts
