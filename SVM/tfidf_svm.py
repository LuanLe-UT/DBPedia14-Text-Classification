import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import load_dataset
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC


OUTPUT_DIR = Path(__file__).resolve().parent / "results_dbpedia_svm"


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

    print(f"Training documents: {len(train_texts)}")
    print(f"Test documents: {len(test_texts)}")
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        token_pattern=r"(?u)\b[^\W_]+\b",
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float32,
    )
    print("Fitting scikit-learn TfidfVectorizer...")
    train_features = vectorizer.fit_transform(train_texts)
    vocabulary_size = len(vectorizer.vocabulary_)
    print(
        f"Vocabulary size: {vocabulary_size:,}; "
        f"TF-IDF training matrix: {train_features.shape[0]:,} x "
        f"{train_features.shape[1]:,} "
        f"({train_features.nnz:,} non-zero values)"
    )

    print("Transforming test data...")
    test_features = vectorizer.transform(test_texts)

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
            "implementation": "sklearn TfidfVectorizer",
            "max_features": max_features,
            "min_df": min_df,
            "vocabulary_size": vocabulary_size,
            "token_pattern": r"(?u)\b[^\W_]+\b",
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True,
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
        description="Classify DBpedia-14 with TF-IDF and LinearSVC."
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
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("--train-limit must be greater than zero.")
    if args.test_limit is not None and args.test_limit < 1:
        parser.error("--test-limit must be greater than zero.")
    if args.max_features < 1 or args.min_df < 1:
        parser.error("--max-features and --min-df must be positive.")
    return args


def main() -> None:
    args = parse_args()
    run(
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        max_features=args.max_features,
        min_df=args.min_df,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
