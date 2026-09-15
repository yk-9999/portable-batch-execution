# Portable Batch Execution

Portable Batch Execution is a backend-agnostic execution framework for finite,
reproducible, CPU-heavy batch workloads.

Its intended architecture consists of a Common Execution Kernel, Domain Packs,
a Trusted Private Project Adapter, and pluggable Execution Backends. Those
components are not implemented in this bootstrap repository yet.

## Non-goals

- Arbitrary shell execution
- Arbitrary Python source execution
- Arbitrary SQL supplied through a JobSpec
- Daemon or real-time workloads
- Storing private project data, models, or results in this public repository
