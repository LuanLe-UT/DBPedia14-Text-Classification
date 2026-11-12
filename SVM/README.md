# DBpedia-14: custom TF-IDF + Linear SVM

This experiment uses the same `content` field and Hugging Face dataset as the
BERT and GPT-2 experiments: `fancyzhx/dbpedia_14`. The implementation includes
its own Unicode word tokenizer, document-frequency vocabulary, sublinear term
frequency, smoothed IDF, and L2 normalization. `LinearSVC` is used for the
linear SVM classifier.

## Run

From the project root, activate the existing Python environment and run:

```powershell
python SVM\tfidf_svm.py
```

The full run uses all 560,000 training records and 70,000 test records. On the
first run, Hugging Face `datasets` downloads the dataset if it is not cached.
For a quick smoke test (the selected samples are balanced across labels):

```powershell
python SVM\tfidf_svm.py --train-limit 2000 --test-limit 500
```

Optional controls include `--max-features`, `--min-df`, `--batch-size`, and
`--output-dir`. The default maximum vocabulary size is 100,000, with terms
appearing in fewer than two training documents excluded.

## Evaluation output

The script prints accuracy, macro precision, macro recall, and macro F1 for
both the training and held-out test sets. It writes:

- `results_dbpedia_svm/metrics.json`: configuration, sample counts, runtime,
  train metrics, and test metrics.
- `results_dbpedia_svm/linear_svc_tfidf.joblib`: fitted vectorizer and
  classifier.
- `results_dbpedia_svm/classification_report_test.txt`: per-class precision,
  recall, F1, and support, plus macro and weighted averages.
- `results_dbpedia_svm/confusion_matrix_test.csv`: test-set confusion matrix.
- `results_dbpedia_svm/confusion_matrix_test.png`: annotated heatmap of the
  test-set confusion matrix.

Metrics from a limited smoke-test run are not comparable to the full-dataset
results. The training metrics are measured on the same examples used to fit
the classifier; test metrics use the separate DBpedia test split.
