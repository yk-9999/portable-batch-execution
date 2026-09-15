from typing import Protocol

from portable_batch_execution.contracts import (
    AdapterDescriptor,
    JobSpec,
    ShardAttemptRecord,
    ShardSpec,
)


class ProjectAdapter(Protocol):
    @property
    def descriptor(self) -> AdapterDescriptor: ...
    def validate_job(self, job: JobSpec) -> None: ...
    def execute(
        self, job: JobSpec, shard: ShardSpec, params: dict, context: object
    ) -> ShardAttemptRecord: ...
    def finalize(
        self,
        job: JobSpec,
        canonical_attempts: tuple[ShardAttemptRecord, ...],
        context: object,
    ) -> object: ...
