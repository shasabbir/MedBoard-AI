# Release acceptance

Phase 10 closes the project against the Definition of Done and architecture checklist in
`plan.md`. Acceptance is executable, not a manual claim.

## Run the release gate

```powershell
python scripts/acceptance.py
```

The gate runs formatting checks, strict typing, all unit/integration/UI tests with the coverage
threshold, dependency validation, the deterministic evaluation suite, and a headless Streamlit
HTTP startup check. Evaluation is rerun and compared by SHA-256. The gate exits nonzero on the
first failure.

## Covered acceptance paths

- concurrent base-agent fan-out and reducer-safe shared state;
- evidence-driven zero/one/multiple specialist routing with stored reasons;
- source-attributed RAG and explicit RAG-outage behavior;
- bounded critic revision and revision-limit behavior;
- deterministic lab, RAG, and risk tool telemetry;
- automatic agent retry exhaustion with visible error and attempt count;
- persisted human interrupt, add-information rerun, second interrupt, approval, final report,
  feedback history, and later case reload;
- rejection without report and retained audit history;
- OpenAI and Gemini structured provider adapters tested through injected clients;
- rendered architecture, messages, trace, errors, timing, tokens, cost, logs, knowledge,
  settings, memory, review controls, and approval-gated report in Streamlit.

## Safety acceptance

The suite asserts that Risk/Triage cannot generate a final report, only approval reaches the
Report Agent, the mandatory clinician-review disclaimer cannot be changed, retrieved evidence
cannot be fabricated during a RAG outage, and state snapshots reject dangling evidence or
hypothesis references.

## Remaining production limitations

Passing this gate establishes reproducible prototype behavior, not clinical validation.
Production use would still require clinician-led external evaluation, privacy/security review,
regulated quality processes, calibrated provider pricing, provider-specific cancellation testing,
and operational monitoring. Live API tests, including timeout wiring, are intentionally mocked in CI to avoid credentials,
cost, nondeterminism, and disclosure of synthetic case content to external services.
