# MedBoard AI Evaluation Results

Benchmark version: `1.0`

These are deterministic offline-demo results. The single-pass configuration is a controlled proxy, not a measured production LLM baseline.

## Capability and ablation metrics

| Configuration | Top-1 / 3 / 5 | Routing P / R | Red flags R / FA | Triage accuracy | Missing-info recall | Unsupported claims |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_pass_proxy | 80.0% / 100.0% / 100.0% | 100.0% / 0.0% | 0.0% / 0 | 20.0% | 100.0% | 100.0% |
| multi_agent_no_critic | 80.0% / 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 0 | 100.0% | 100.0% | 0.0% |
| full_medboard | 80.0% / 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 0 | 100.0% | 100.0% | 0.0% |

## Efficiency metrics

| Configuration | Mean approximate tokens | Mean agent calls | Mean revisions |
| --- | ---: | ---: | ---: |
| single_pass_proxy | 148.4 | 1.0 | 0.0 |
| multi_agent_no_critic | 2768.8 | 7.8 | 0.0 |
| full_medboard | 3000.6 | 10.2 | 0.4 |

## RAG retrieval

- Recall@K: 100.0%
- Mean reciprocal rank: 1.000
- Citation metadata completeness: 100.0%

## Metric definitions

- Differential recall checks whether each labeled consideration appears within the first 1, 3, or 5 ranked outputs.
- Routing precision and recall compare selected specialist-case pairs with their labeled expectations.
- Unsupported-claim rate is the share of differential considerations without structured supporting-evidence IDs.
- RAG relevance uses the labeled source document; citation completeness requires document, organization, section, URL, and chunk ID.

## Interpretation limits

- The benchmark uses small, synthetic cases and tests software behavior, not clinical efficacy.
- Expected diagnoses are differential considerations, never autonomous diagnoses.
- Demo token counts are approximate and demo cost is zero.
- Human clinical review remains mandatory for every generated report.
