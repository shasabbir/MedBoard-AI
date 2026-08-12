# Five-minute demonstration script

## 0:00–0:40 — Frame the project

Open the New case page. Point out `DEMO` mode and the safety notice. Explain that this is an
educational decision-support board: the visible product is collaborative investigation under
human control, not an autonomous diagnosis.

## 0:40–1:30 — Start a dynamic case

Select **Neurological** and start the review. While it runs, explain that History, Symptoms,
Laboratory, and Medication are parallel LangGraph branches writing structured evidence into
shared state.

## 1:30–2:20 — Prove orchestration

On Workflow, show the rendered graph and agent cards. Highlight that Neurology is complete,
while Cardiology and Infectious Disease are visibly not selected. Use the metrics to show
completed calls, timing, messages, tokens, and cost.

## 2:20–3:10 — Inspect collaboration

Open Analysis and identify multiple differential considerations, evidence IDs, specialist
routing reasons, and missing information. Open Messages to show requests/responses rather
than a hidden single prompt. Open Evidence to show the stroke source, section, excerpt,
similarity score, and public URL.

## 3:10–4:00 — Show criticism and safety

Open Trace to show the execution order and clean error panel. Explain that the red-team critic
can request bounded revisions. Highlight the emergency triage banner and clarify that the Risk
Agent expresses urgency but never generates the final report.

## 4:00–4:35 — Demonstrate human control

Open Human review. Select **approve**, add a reviewer identifier, and submit. Open Report and
show that it exists only after approval and includes the mandatory experimental-system
disclaimer. Mention that reject, add information, revision, specialist, and retry paths are
also persisted actions.

## 4:35–5:00 — Close with persistence and evaluation

Open Saved cases and reload the run. Briefly show Knowledge base and System logs. Finish with
the checked-in evaluation: five labeled cases, three ablations, differential/routing/safety
metrics, and four RAG questions. State clearly that the benchmark validates repeatable
software behavior—not clinical efficacy.

## Backup commands

If browser display is unavailable, use:

```powershell
python -m medboard.cli --case data/demo_cases/neurological.json
python -m medboard.evaluation --output evaluation/results
```
