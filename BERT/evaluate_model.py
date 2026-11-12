from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_dbpedia"


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["content"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main():
    checkpoint_path = get_last_checkpoint(str(OUTPUT_DIR))
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No training checkpoint found in '{OUTPUT_DIR}'. "
            "Train the model first or add a checkpoint to this directory."
        )

    dataset = load_dataset("fancyzhx/dbpedia_14")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_dataset = dataset["test"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "evaluation"),
            per_device_eval_batch_size=8,
            report_to="none",
        ),
        compute_metrics=compute_metrics,
    )
    predictions = trainer.predict(eval_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    class_names = dataset["train"].features["label"].names

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of test samples: {len(eval_dataset)}")
    print("\nKết quả đánh giá (macro average):")
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        print(f"{metric_name.capitalize()}: {predictions.metrics[f'test_{metric_name}']:.4f}")

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps({
            "dataset": "fancyzhx/dbpedia_14",
            "model": MODEL_NAME,
            "checkpoint": str(checkpoint_path),
            "sample_counts": {"test": len(eval_dataset)},
            "test": {name: float(predictions.metrics[f"test_{name}"]) for name in ("accuracy", "precision", "recall", "f1")},
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "confusion_matrix_test.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])
    (OUTPUT_DIR / "classification_report_test.txt").write_text(
        classification_report(true_labels, predicted_labels, labels=range(len(class_names)), target_names=class_names, zero_division=0),
        encoding="utf-8",
    )
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - DBpedia Test Set ({len(eval_dataset)} samples)")
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_test.png", dpi=200, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()

