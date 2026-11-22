# RAG answer scores

Reader: `overlap`. Split: `test`. Questions with a clean passage: 50.

Delta is the noisy score minus the clean score on the same questions. A negative delta means the noise lowered the score.

| Condition | N | Exact Match | Token F1 | Delta EM | Delta F1 | Mean context tokens | Mean seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 50 | 0.00% | 10.09% |  |  | 133.5 | 0.000 |
| Semantic Noise | 50 | 0.00% | 9.80% | +0.00 pp | -0.29 pp | 234.7 | 0.000 |
| Datatype Noise | 50 | 0.00% | 10.19% | +0.00 pp | +0.10 pp | 280.7 | 0.000 |
| Illegal-Sentence Noise | 50 | 0.00% | 10.09% | +0.00 pp | +0.00 pp | 163.5 | 0.000 |
| Counterfactual Noise | 50 | 0.00% | 7.34% | +0.00 pp | -2.75 pp | 168.6 | 0.000 |
| Supportive Noise | 50 | 0.00% | 8.70% | +0.00 pp | -1.39 pp | 234.7 | 0.000 |
| Orthographic Noise | 50 | 0.00% | 10.09% | +0.00 pp | +0.00 pp | 256.3 | 0.000 |
