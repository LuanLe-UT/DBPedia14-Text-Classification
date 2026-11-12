import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from datasets import load_dataset
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
OUTPUT_DIR = Path(__file__).resolve().parent / "results_dbpedia_svm"


class TfidfVectorizer:
    """A sparse, word-level TF-IDF vectorizer implemented without sklearn."""

    def __init__(self, max_features: int = 100_000, min_df: int = 2):
        if max_features < 1:
            raise ValueError("max_features must be greater than zero.")
        if min_df < 1:
            raise ValueError("min_df must be greater than zero.")
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def fit(self, documents: Iterable[str]) -> "TfidfVectorizer":
        document_frequency: Counter[str] = Counter()
        document_count = 0
        for document in documents:
            document_count += 1
            document_frequency.update(set(self.tokenize(document)))

        if document_count == 0:
            raise ValueError("Cannot fit TF-IDF on an empty document collection.")

        terms = [
            (term, frequency)
            for term, frequency in document_frequency.items()
            if frequency >= self.min_df
        ]
        terms.sort(key=lambda item: (-item[1], item[0]))
        terms = terms[: self.max_features]
        if not terms:
            raise ValueError(
                "The vocabulary is empty; lower min_df or provide more documents."
            )

        self.vocabulary_ = {
            term: index for index, (term, _) in enumerate(terms)
        }
        self.idf_ = np.asarray(
            [
                math.log((1 + document_count) / (1 + frequency)) + 1
                for _, frequency in terms
            ],
            dtype=np.float64,
        )
        return self

    def transform(
        self, documents: Sequence[str], batch_size: int = 10_000
    ) -> csr_matrix:
        if self.idf_ is None:
            raise RuntimeError("Call fit before transform.")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero.")
        if not documents:
            return csr_matrix((0, len(self.vocabulary_)), dtype=np.float32)

        batches = []
        for start in range(0, len(documents), batch_size):
            batch_documents = documents[start : start + batch_size]
            indptr = [0]
            indices: list[int] = []
            values: list[float] = []

            for document in batch_documents:
                term_counts = Counter(self.tokenize(document))
                row: list[tuple[int, float]] = []
                for term, count in term_counts.items():
                    feature_index = self.vocabulary_.get(term)
                    if feature_index is not None:
                        term_frequency = 1.0 + math.log(count)
                        row.append(
                            (
                                feature_index,
                                term_frequency * self.idf_[feature_index],
                            )
                        )

                row.sort(key=lambda item: item[0])
                norm = math.sqrt(sum(value * value for _, value in row))
                if norm:
                    indices.extend(index for index, _ in row)
                    values.extend(value / norm for _, value in row)
                indptr.append(len(indices))

            batch = csr_matrix(
                (
                    np.asarray(values, dtype=np.float32),
                    np.asarray(indices, dtype=np.int32),
                    np.asarray(indptr, dtype=np.int32),
                ),
                shape=(len(batch_documents), len(self.vocabulary_)),
                dtype=np.float32,
            )
            batches.append(batch)

        return vstack(batches, format="csr", dtype=np.float32)

    def fit_transform(
        self, documents: Sequence[str], batch_size: int = 10_000
    ) -> csr_matrix:
        self.fit(documents)
        return self.transform(documents, batch_size=batch_size)


def evaluate_split(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def balanced_indices(labels: Sequence[int], sample_limit: int) -> list[int]:
    class_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        class_indices.setdefault(int(label), []).append(index)

    classes = sorted(class_indices)
    if sample_limit >= len(labels):
        return list(range(len(labels)))

    per_class, remainder = divmod(sample_limit, len(classes))
    selected = []
    for class_position, label in enumerate(classes):
        count = per_class + (class_position < remainder)
        selected.extend(class_indices[label][:count])
    return sorted(selected)


def write_confusion_matrix(
    matrix: np.ndarray, class_names: Sequence[str], destination: Path
) -> None:
    with destination.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: Sequence[str],
    destination: Path,
    sample_count: int,
) -> None:
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - DBpedia Test Set ({sample_count} samples)")
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close()


def run(
    train_limit: int | None,
    test_limit: int | None,
    max_features: int,
    min_df: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    dataset = load_dataset("fancyzhx/dbpedia_14")
    train_data = dataset["train"]
    test_data = dataset["test"]
    class_names = dataset["train"].features["label"].names
    if train_limit is not None:
        train_data = train_data.select(
            balanced_indices(
                train_data["label"], min(train_limit, len(train_data))
            )
        )
    if test_limit is not None:
        test_data = test_data.select(
            balanced_indices(test_data["label"], min(test_limit, len(test_data)))
        )

    train_texts = train_data["content"]
    test_texts = test_data["content"]
    train_labels = np.asarray(train_data["label"])
    test_labels = np.asarray(test_data["label"])

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    print(f"Training documents: {len(train_texts)}")
    print(f"Test documents: {len(test_texts)}")
    started = time.perf_counter()
    print("Fitting custom tokenizer and TF-IDF vocabulary...")
    train_features = vectorizer.fit_transform(train_texts, batch_size=batch_size)
    print(
        f"Vocabulary size: {len(vectorizer.vocabulary_):,}; "
        f"TF-IDF training matrix: {train_features.shape[0]:,} x "
        f"{train_features.shape[1]:,} "
        f"({train_features.nnz:,} non-zero values)"
    )

    print("Transforming test data...")
    test_features = vectorizer.transform(test_texts, batch_size=batch_size)

    print("Training LinearSVC...")
    classifier = LinearSVC(random_state=42, max_iter=5000)
    classifier.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - started

    print("Evaluating train and test splits...")
    train_predictions = classifier.predict(train_features)
    test_predictions = classifier.predict(test_features)
    metrics = {
        "dataset": "fancyzhx/dbpedia_14",
        "model": "LinearSVC",
        "vectorizer": {
            "implementation": "custom word tokenizer + sublinear TF-IDF",
            "max_features": max_features,
            "min_df": min_df,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "normalization": "L2",
        },
        "sample_counts": {
            "train": len(train_texts),
            "test": len(test_texts),
        },
        "training_and_vectorization_seconds": round(training_seconds, 2),
        "train": evaluate_split(train_labels, train_predictions),
        "test": evaluate_split(test_labels, test_predictions),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "linear_svc_tfidf.joblib"
    dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "class_names": class_names,
        },
        model_path,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report_test.txt").write_text(
        classification_report(
            test_labels,
            test_predictions,
            labels=range(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    write_confusion_matrix(
        confusion_matrix(
            test_labels, test_predictions, labels=range(len(class_names))
        ),
        class_names,
        output_dir / "confusion_matrix_test.csv",
    )
    save_confusion_matrix_plot(
        confusion_matrix(
            test_labels, test_predictions, labels=range(len(class_names))
        ),
        class_names,
        output_dir / "confusion_matrix_test.png",
        len(test_labels),
    )

    print("\nTrain metrics (macro average):")
    for metric, value in metrics["train"].items():
        print(f"  {metric}: {value:.4f}")
    print("\nTest metrics (macro average):")
    for metric, value in metrics["test"].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nTraining + vectorization time: {training_seconds:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Trained model saved to: {model_path}")
    print(f"Per-class test report: {output_dir / 'classification_report_test.txt'}")
    print(f"Confusion matrix plot: {output_dir / 'confusion_matrix_test.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify DBpedia-14 with a custom TF-IDF and LinearSVC."
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        help="Use only the first N training records (for quick experiments).",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        help="Use only the first N test records (for quick experiments).",
    )
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("--train-limit must be greater than zero.")
    if args.test_limit is not None and args.test_limit < 1:
        parser.error("--test-limit must be greater than zero.")
    if args.max_features < 1 or args.min_df < 1 or args.batch_size < 1:
        parser.error("--max-features, --min-df, and --batch-size must be positive.")
    return args


def main() -> None:
    args = parse_args()
    run(
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        max_features=args.max_features,
        min_df=args.min_df,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
