# MedBoard AI

*Interactive Multi-Agent Medical Diagnostic Decision Support Board*

## Project Overview

MedBoard AI is an interactive multi-agent system for educational clinical reasoning and medical decision support. A Supervisor Agent coordinates focused agents that examine patient history, symptoms, laboratory results, medications, competing differential diagnoses, specialist perspectives, and supporting medical evidence.

The system works as a multidisciplinary case-review board. Agents share structured evidence, challenge weak conclusions, request missing information, revise the analysis when necessary, assess urgency, and pause for clinician review before producing a final report. It is a decision-support prototype, not an autonomous diagnostic or prescribing system.

![MedBoard AI multi-agent workflow](../../diagram_final.png)

## Agent Roles

| Agent | Primary role |
|---|---|
| Supervisor | Plans the investigation, routes work, coordinates revisions, and handles failures. |
| Patient History | Extracts history, risk factors, negative findings, and missing context. |
| Symptom Analysis | Normalizes symptoms, identifies clinical patterns, and flags red-flag combinations. |
| Laboratory Analysis | Validates values and units, detects abnormalities, and identifies missing tests. |
| Medication | Reviews adverse-effect relevance, interactions, duplication, and medication-related concerns. |
| Differential Diagnosis | Integrates evidence into multiple competing, traceable hypotheses. |
| Specialist Agents | Cardiology, Neurology, and Infectious Disease agents are selected only when relevant. |
| Evidence Retrieval / RAG | Retrieves concise, source-backed medical evidence from the knowledge store. |
| Red-Team Critic | Challenges assumptions, contradictions, unsupported claims, and premature closure. |
| Risk / Triage | Detects urgency and escalation needs without making the final diagnosis. |
| Final Report Generator | Produces the structured report after human approval. |

## Technology Stack

| Area | Technology |
|---|---|
| Agent orchestration | LangGraph with LangChain where useful |
| Models and schemas | OpenAI or Google Gemini through a central provider layer; Pydantic validation |
| Backend | Python 3.11+ |
| Streamlit UI | Interactive case input, live workflow, messages, memory, evidence, human controls, and reports |
| Workflow memory | Structured LangGraph state and checkpointing |
| Knowledge and RAG | ChromaDB or FAISS with metadata-backed medical references |
| Persistent memory | SQLite for cases, runs, messages, feedback, retrievals, and errors |
| Observability and testing | Structured logging, trace and cost tracking, retries, and Pytest |

## Main Features

- Live agent status, execution trace, and interactive workflow graph
- Structured claims, evidence IDs, contradictions, and agent communication history
- Parallel first-line analysis with dynamic specialist routing
- Multiple competing differential considerations with visible uncertainty
- Source-backed RAG retrieval with inspectable excerpts and metadata
- Red-team criticism and a bounded revision loop
- Deterministic red-flag checks and risk/triage assessment
- Human pause, add-information, specialist-request, approve, reject, and retry controls
- Workflow, knowledge, and searchable case-history memory
- Token usage, estimated API cost, timing, logs, retries, and error reporting

## Originality and Expected Outcome

MedBoard AI differs from a general medical chatbot by making collaborative reasoning visible. Specialized agents build structured evidence, a dynamic router selects relevant expertise, RAG grounds the review, a critic challenges the result, and a human clinician controls the final decision.

The completed system will demonstrate an explainable AI workforce that can investigate synthetic or de-identified cases, communicate findings, preserve disagreements, recover from partial failures, and produce an auditable case-analysis report through an interactive dashboard.

## Safety Scope

MedBoard AI is an experimental university decision-support prototype. It does not provide definitive diagnoses or prescriptions, and every generated report requires review by a qualified healthcare professional.
