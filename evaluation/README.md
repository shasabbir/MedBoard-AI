# Evaluation methodology

MedBoard's evaluation suite is a deterministic, reproducible software benchmark. It is
designed to test whether the multi-agent workflow exhibits its intended capabilities; it
does not measure clinical efficacy or validate the system as a medical device.

## Reproduce the results

From the repository root, run:

```powershell
python -m medboard.evaluation --output evaluation/results
```

The command reads the versioned labels in `data/benchmarks`, runs every configuration over
the same synthetic cases, evaluates the local knowledge store, and emits JSON plus Markdown.
No API key or network connection is needed.

## Benchmark coverage

The five cases cover:

- iron-deficiency anemia pattern with urgent low-hemoglobin flag;
- focal neurological emergency;
- febrile respiratory presentation;
- acute chest-pain emergency;
- a routine nonspecific presentation that checks false alarms and empty routing.

Each case labels expected differential text, justified specialists, red-flag presence,
triage level, and important missing information. Four RAG questions label the relevant
source document for anemia, chest pain, pneumonia, and stroke.

## Metrics

- Differential Top-1/3/5 recall is micro-averaged over labeled considerations.
- Routing precision/recall is calculated over specialist-case pairs. Selecting no specialist
  when none is expected is correct, but contributes no true-positive pair.
- Red-flag recall is calculated over positive cases; false alarms are counted on negative
  cases. Triage accuracy requires the exact labeled level.
- Missing-information recall is micro-averaged over labeled requests with normalized,
  case-insensitive text matching.
- Unsupported-claim rate counts differential considerations with no supporting evidence IDs.
- RAG Recall@K and mean reciprocal rank use the labeled document. Citation completeness
  requires document, organization, section, source URL, and chunk ID.
- Agent calls count completed agent executions, including critic-driven reruns. Demo token
  counts are approximate and demo API cost is zero.

## Controlled configurations

1. `single_pass_proxy` is a deterministic one-pass baseline over raw case fields. It is a
   software ablation proxy, not a measured single-LLM baseline.
2. `multi_agent_no_critic` retains parallel intake, dynamic routing, specialists, and RAG,
   but removes the red-team critic/revision loop. Deterministic risk rules are evaluated
   separately so red-flag metrics remain comparable.
3. `full_medboard` includes the complete collaboration graph, RAG, bounded critic loop,
   and risk/triage stage.

## Limitations

The dataset is deliberately small and synthetic, and the deterministic provider is optimized
for repeatable demonstrations. The reported values should therefore be interpreted as
regression evidence for architecture and workflow behavior—not estimates of diagnostic
accuracy, generalization, safety, or real-world clinical performance. A future live-provider
study needs a larger externally reviewed benchmark, repeated runs, confidence intervals,
and clinician-led error analysis.
