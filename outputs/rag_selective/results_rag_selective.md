# Detector selection versus unfiltered retrieval

Reader: `extractive`. Split: `test`. Questions: 50.

`retrieved` is BM25 top-k with every stored passage, including harmful noise.
`detector_selective` drops passages predicted as counterfactual, supportive, or orthographic noise,
then places predicted Relevant passages first and beneficial noise (semantic, datatype, illegal sentence) below them.
`clean` is the relevant passage alone.

Delta is the score minus the unfiltered `retrieved` score on the same questions.
A positive delta means the detector context scored higher than leaving harmful noise in.

| Condition | N | Exact Match | Token F1 | Delta EM | Delta F1 | Mean passages | Harmful kept | Beneficial kept | Relevant first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retrieved | 50 | 22.00% | 32.61% | +0.00 pp | +0.00 pp | 7.00 | 3.00 | 3.00 | 18.00% |
| detector_selective | 50 | 62.00% | 76.15% | +40.00 pp | +43.54 pp | 4.02 | 0.10 | 2.96 | 92.00% |
| clean | 50 | 60.00% | 74.87% | +38.00 pp | +42.27 pp | 1.00 | 0.00 | 0.00 | 100.00% |
