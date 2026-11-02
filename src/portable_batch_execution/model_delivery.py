"""Validated, transport-only Hugging Face private model delivery descriptor.

Model content identity remains the four pinned static ArtifactRefs. The source
is not a client-supplied operation parameter or part of the request fingerprint.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


class HFPrivateModelDelivery(BaseModel):
    """One private model repository pinned to one immutable Git commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["hf-private-repo-oidc"]
    repo_id: str
    revision: str
    filenames: tuple[str, str, str, str]

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        if not _REPO_ID.fullmatch(value) or ".." in value:
            raise ValueError("model delivery repository must be a safe namespace/repo")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not _REVISION.fullmatch(value):
            raise ValueError("model delivery requires a 40-character commit SHA")
        return value

    @field_validator("filenames")
    @classmethod
    def validate_filenames(
        cls, values: tuple[str, str, str, str]
    ) -> tuple[str, str, str, str]:
        for name in values:
            parts = name.split("/")
            if (
                len(name) > 255
                or not 2 <= len(parts) <= 8
                or any(not _COMPONENT.fullmatch(part) or part in {".", ".."} for part in parts)
                or "\\" in name
            ):
                raise ValueError("invalid model delivery relative filename")
        if len(set(values)) != 4:
            raise ValueError("model delivery filenames must be unique")
        if tuple(name.split("/")[-1] for name in values) != (
            "config.json", "model.safetensors", "config.json", "model.safetensors"
        ):
            raise ValueError("model delivery file order is invalid")
        return values

    @model_validator(mode="after")
    def distinct_model_directories(self):
        if self.filenames[0].rsplit("/", 1)[0] == self.filenames[2].rsplit("/", 1)[0]:
            raise ValueError("model A and B must use different directories")
        if self.filenames[0].rsplit("/", 1)[0] != self.filenames[1].rsplit("/", 1)[0]:
            raise ValueError("model A config and weights must share a directory")
        if self.filenames[2].rsplit("/", 1)[0] != self.filenames[3].rsplit("/", 1)[0]:
            raise ValueError("model B config and weights must share a directory")
        return self
