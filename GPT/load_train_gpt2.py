from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# GPT-2 Small (124M parameters), Decoder-only Transformer
MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_dbpedia_gpt2"

NUM_LABELS = 14
MAX_LENGTH = 128


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["content"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def main():
    dataset = load_dataset("fancyzhx/dbpedia_14")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 does not define a separate PAD token.
    # Reuse EOS as PAD for batched sequence classification.
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        desc="Tokenizing DBpedia-14 training data",
    )

    class_names = dataset["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Number of labels: {NUM_LABELS}")
    print("Starting fine-tuning...")

    train_result = trainer.train()

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\nFine-tuning completed.")
    print(f"Model and tokenizer saved to: {OUTPUT_DIR}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training steps: {train_result.global_step}")


if __name__ == "__main__":
    main()
