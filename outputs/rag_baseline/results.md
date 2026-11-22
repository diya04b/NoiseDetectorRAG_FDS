# RAG answer scores

Reader: `extractive`. Split: `test`. Questions with a clean passage: 50.

Delta is the noisy score minus the clean score on the same questions. A negative delta means the noise lowered the score.

| Condition | N | Exact Match | Token F1 | Delta EM | Delta F1 | Mean context tokens | Mean seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 50 | 60.00% | 74.87% |  |  | 133.5 | 1.607 |
| Semantic Noise | 50 | 58.00% | 72.45% | -2.00 pp | -2.42 pp | 234.7 | 1.614 |
| Datatype Noise | 50 | 52.00% | 69.26% | -8.00 pp | -5.61 pp | 280.7 | 1.588 |
| Illegal-Sentence Noise | 50 | 58.00% | 74.05% | -2.00 pp | -0.82 pp | 163.5 | 1.599 |
| Counterfactual Noise | 50 | 10.00% | 21.55% | -50.00 pp | -53.32 pp | 168.6 | 4.616 |
| Supportive Noise | 50 | 56.00% | 73.62% | -4.00 pp | -1.26 pp | 234.7 | 4.263 |
| Orthographic Noise | 50 | 56.00% | 68.59% | -4.00 pp | -6.28 pp | 256.3 | 1.624 |
