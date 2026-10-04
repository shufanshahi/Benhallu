# অলীকবচন: Bengali Hallucination Detection

I built a Bengali hallucination detection pipeline that checks whether a response is supported by a given passage or by a retrieved answer from a Bengali question-answer knowledge base. The implementation is in [kaggle_hallucination_pipeline.ipynb](kaggle_hallucination_pipeline.ipynb), which creates the `hallu/` Python package when run in Kaggle.

## Dataset preparation

I collected datasets and preprocessed them to create my own Bengali hallucination dataset for fine-tuning a model. This data-preparation work is separate from the notebook's detection pipeline; the notebook uses labelled samples to train its decision classifiers and does not run a model fine-tuning step.

For the detection pipeline, I clean and normalize Bengali question-answer pairs, remove empty and duplicate records, and build a searchable knowledge base. The pipeline also excludes knowledge-base questions that match labelled sample questions to reduce validation leakage. The input datasets are attached to the Kaggle notebook rather than stored in this repository.

## Pipeline

The pipeline routes each row according to whether it has a usable `context` passage:

1. **With context:** Split the passage into sentence windows and retrieve relevant evidence using Bengali word and character TF-IDF signals. An optional Bengali extractive QA model proposes an answer from the evidence. Compare that answer with the candidate response using lexical, numeric, answer-type, and other agreement features, then classify it with a scaled logistic regression model.
2. **Without context:** Retrieve a relevant question-answer pair from the knowledge base using inverted indexes, BM25, fuzzy matching, entity overlap, and optional LaBSE embeddings. Compare the candidate response with the retrieved answer, then classify it with a second scaled logistic regression model. Low-confidence retrievals use a fallback label.

Both routes share Bengali text and number normalization, including Bengali digits, number words, and ordinals. The no-context route tunes retrieval weights and its decision threshold on labelled samples using grouped cross-validation and macro-F1. Transformer-based components are optional; the notebook can fall back to lexical features when their weights are unavailable.

## Run on Kaggle

Attach these inputs to the notebook:

- `test set.csv` — test rows with `id`, `prompt_bn`, `response_bn`, and `context`.
- `dataset samples.json` — labelled sample rows used to fit and validate the classifiers.
- `combined_bengali_qa.csv` — Bengali question-answer knowledge base with `prompt_bn` and `response_bn`.

Run [kaggle_hallucination_pipeline.ipynb](kaggle_hallucination_pipeline.ipynb) from top to bottom. It installs `rank_bm25` and `rapidfuzz`, writes the `hallu/` modules, detects the attached files under `/kaggle/input`, builds or loads cached indexes, and writes `/kaggle/working/submission.csv`. The output contains `id` and `label`, where `1` means faithful and `0` means hallucinated. The last cell checks row count, unique IDs, and valid labels.
