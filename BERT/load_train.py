from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_dbpedia"


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["content"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def main():
    dataset = load_dataset("fancyzhx/dbpedia_14")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=14,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        fp16=True,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
