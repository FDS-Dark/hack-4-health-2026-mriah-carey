import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import torch

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "merged_plain_language_dataset.csv"
MODEL_NAME = "razent/SciFive-large-PubMed"
OUTPUT_DIR = "./scifive-lora-merged"
SOURCE_COL = "original_text"
TARGET_COL = "plain_language_text"
MAX_SOURCE_LEN = 768
MAX_TARGET_LEN = 256
GRADE_TOKEN = "<grade6>"

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(CSV_PATH)

# create HF datasets grouped by split column
dataset_dict = {}
for split_name in df["split"].unique():
    subset = df[df["split"] == split_name]
    dataset_dict[split_name] = Dataset.from_pandas(subset)

raw_datasets = DatasetDict(dataset_dict)

# -----------------------
# LOAD MODEL & TOKENIZER
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if GRADE_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([GRADE_TOKEN])

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model.resize_token_embeddings(len(tokenizer))

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q", "v"],
)
model = get_peft_model(model, peft_config)

# -----------------------
# PROMPT BUILDER
# -----------------------
def add_prompt(example):
    instruction = (
        f"{GRADE_TOKEN} Simplify the biomedical text for a 6th-grade reader, "
        "preserving facts, uncertainty, and measurements without adding medical advice."
    )
    src = example[SOURCE_COL]
    tgt = example[TARGET_COL]

    example["input_text"] = f"{instruction}\n\nText:\n{src}\n\nSimplified:"
    example["target_text"] = tgt
    return example

processed = raw_datasets.map(add_prompt)

# -----------------------
# TOKENIZATION
# -----------------------
def tokenize(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )
    labels_ids = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs

tokenized = processed.map(tokenize, batched=True, remove_columns=processed["train"].column_names)

# -----------------------
# METRICS
# -----------------------
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [
        tokenizer.decode([id if id != -100 else tokenizer.pad_token_id for id in l], skip_special_tokens=True)
        for l in labels
    ]
    return rouge.compute(predictions=preds, references=labels)

# -----------------------
# TRAINING SETUP
# -----------------------
training_args = Seq2SeqTrainingArguments(
    OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("validation") or None,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

