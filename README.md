# NoiseDetectorRAG

Noise detection and pairwise noise-interaction analysis for retrieval-augmented generation.

The dataset milestone builds a documented eight-class question–passage dataset from the released NoiserBench files. The detector milestone fine-tunes RoBERTa-base on that dataset and scores the held-out test split. The selection milestone ranks each question's passages, then removes or reranks them with the detector labels and writes the final passage list. It does not modify the NoiserBench repository, and it does not generate an answer.

For the dataset files and commands, read `docs/FILE_GUIDE.md`. For the detector files and commands, read `docs/DETECTOR_GUIDE.md`. For the remove and rerank step, read `docs/SELECTION_GUIDE.md`.

## Show the NoiserBench issues

`docs/noiserbench_issues.md` is the note for the course report. Regenerate it with:

```bash
git clone --depth 1 https://github.com/jinyangwu/NoiserBench.git /tmp/NoiserBench
python scripts/download_noiserbench.py
python scripts/audit_noiserbench.py --repo /tmp/NoiserBench --data data/raw/noiserbench
```

The audit script is read-only. Twenty-two checks were confirmed against commit `447bee38d91996615ecc85c4da4a5b204f6c9837`.

## Rebuild the derived dataset

```bash
python scripts/download_noiserbench.py
python scripts/inspect_dataset.py
python scripts/run_dataset_pipeline.py
python scripts/validate_dataset.py
python -m unittest tests.test_dataset tests.test_audit_fixture
```

Raw JSON files are downloaded into `data/raw/noiserbench/` and are not committed. The derived JSONL is `data/processed/classification.jsonl`, also untracked. Checksums, reports, and plots are committed.

## What the derived set contains

HotpotQA supplies the seven passage classes. PriorQA supplies Prior Noise only. One passage is kept per parent question and class. Orthographic noise uses ratio 0.2. PriorQA counterfactual passages are held out. The split is 80/10/10 by normalized question, seed 42.

Measured result: 4,000 pairs, 500 per class, with 400/50/50 pairs per class in train/validation/test. Question overlap across splits was 0. See `PROGRESS.md` for the passage-title overlap that this split does not remove.

## Train and test the detector

```bash
python -m pip install matplotlib pyyaml transformers
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/train_detector.py
```

The epoch is chosen by validation macro-F1. The test split is scored after that choice. Results go to `outputs/detector/`. The plain-language map of every detector file is `docs/DETECTOR_GUIDE.md`.

## Select the final passages

```bash
python scripts/select_passages.py
python -m unittest tests.test_selection
```

This reads the saved test predictions, ranks the passages stored for each test question, and writes `outputs/selection/final_retrieved_passages.jsonl`. The plain-language map is `docs/SELECTION_GUIDE.md`.

## Answer the questions

```bash
python -m pip install matplotlib pyyaml transformers
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
py -3.12 scripts/run_rag.py
py -3.12 -m unittest tests.test_rag
```

The reader is a vanilla extractive RAG model, `deepset/roberta-base-squad2`. Run it with Python 3.12, which has PyTorch installed. It answers from the clean passage, from the clean passage plus one noise class, and from the passages the detector kept. Exact Match and token F1 go to `outputs/rag/`. The plain-language map is `docs/RAG_GUIDE.md`. Add `--reader overlap` to run without downloading the reading model. Add `--pairwise` for the two-noise combinations in section 3.9.

```bash
py -3.12 scripts/run_rag.py
py -3.12 scripts/run_rag.py --pairwise
```
