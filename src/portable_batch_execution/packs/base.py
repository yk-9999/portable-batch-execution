from typing import Protocol


class DomainPack(Protocol):
    pack_id: str
    supported_operations: tuple[str, ...]

    def validate_params(self, operation: str, params: dict) -> dict: ...
    def execute(self, job, shard, params, context): ...
    def finalize(self, job, canonical_attempts, context): ...
