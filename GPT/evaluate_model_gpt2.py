from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
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

MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_dbpedia_gpt2"

MAX_LENGTH = 128
BATCH_SIZE = 8


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["content"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
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
        "f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }


def main():
    if not (OUTPUT_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Trained GPT-2 model was not found in '{OUTPUT_DIR}'. "
            "Run load_train_gpt2.py first."
        )

    dataset = load_dataset("fancyzhx/dbpedia_14")

    # Load the tokenizer saved with the fine-tuned model.
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    tokenizer.pad_token = tokenizer.eos_token

    test_dataset = dataset["test"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        desc="Tokenizing DBpedia-14 test data",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(OUTPUT_DIR)
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "evaluation"),
            per_device_eval_batch_size=BATCH_SIZE,
            fp16=__import__("torch").cuda.is_available(),
            report_to="none",
        ),
        compute_metrics=compute_metrics,
    )

    # Final evaluation on the held-out test set.
    test_output = trainer.predict(test_dataset)
    predicted_labels = np.argmax(test_output.predictions, axis=-1)
    true_labels = test_output.label_ids
    class_names = dataset["train"].features["label"].names

    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {OUTPUT_DIR}")
    print(f"Number of test samples: {len(test_dataset)}")

    print("\nKết quả trên tập train (sau fine-tuning):")

    print("\nKết quả đánh giá trên tập test (macro average):")
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        print(
            f"{metric_name.capitalize():<10}: "
            f"{test_output.metrics[f'test_{metric_name}']:.4f}"
        )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps({
            "dataset": "fancyzhx/dbpedia_14",
            "model": MODEL_NAME,
            "checkpoint": str(OUTPUT_DIR),
            "sample_counts": {"test": len(test_dataset)},
            "test": {name: float(test_output.metrics[f"test_{name}"]) for name in ("accuracy", "precision", "recall", "f1")},
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
    plt.title(
        f"Confusion Matrix - GPT-2 / DBpedia Test Set "
        f"({len(test_dataset)} samples)"
    )
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
