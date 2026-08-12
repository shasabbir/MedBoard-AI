# Architecture and design decisions

MedBoard is a stateful investigation graph, not a sequence of role-labeled prompts. The
authoritative diagram is maintained in the root README and rendered in the Workflow tab.

## State and collaboration

`MedicalCaseState` is the shared LangGraph state. Append-only reducers safely merge evidence,
messages, trace events, errors, retrievals, specialist opinions, contradictions, and usage
records from concurrent branches. `MedicalCaseSnapshot` validates referential integrity at
persistence boundaries, including every evidence, hypothesis, question, and retrieval ID.

The Supervisor fans out to History, Symptom, Laboratory, and Medication agents. Their results
join before differential reasoning. The Specialist Router examines structured evidence and
selects only justified specialists; a branch in the diagram represents conditional selection,
not unconditional execution.

## Review and control loops

RAG responds only to explicit questions emitted by the differential and selected specialists.
The critic either accepts progression or returns through a bounded supervisor/differential
revision path. Risk/Triage runs after that gate and cannot create a report. LangGraph then
interrupts at Human Review. Approve alone reaches Report Generator; every other action retains
an audit record and either ends or reruns an affected downstream path.

## Memory boundaries

- Workflow memory is validated in-flight LangGraph state and SQLite checkpoints.
- Knowledge memory is the local source-attributed Chroma vector collection.
- Case-history memory is separate SQLite persistence for snapshots, messages, traces, and
  human feedback.

These boundaries prevent retrieved documents, temporary execution state, and durable case
records from being conflated.

## Observability and safety

Agent execution produces append-only trace, token, error, and message records. The dashboard
shows status, execution time, completed calls, tool calls, approximate demo tokens, cost,
logs, evidence sources, and failure details. Safety language appears before case execution and
inside every approved report. Demo and future live providers share the graph contract.
