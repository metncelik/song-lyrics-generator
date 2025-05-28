import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset
import os
sys.path.append(str(Path(__file__).parent.parent))
from database.client import DatabaseClient
from utils import process_data
from training_config import *

db_client = DatabaseClient()
raw_songs_data = db_client.get_songs()

processed_songs = [process_data(item) for item in raw_songs_data]

songs_dataset = Dataset.from_list(processed_songs)

train_test_split = songs_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.save_pretrained(TOKENIZER_DIR)

def tokenize_function(examples):
    return tokenizer(
        examples["lyrics"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(
    ["lyrics", "song_title", "artist_name"])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(
    ["lyrics", "song_title", "artist_name"])

tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

# Load GPT-2 base model (774M parameters)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Check if we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    logging_dir=LOGGING_DIR,
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    fp16=USE_FP16_ON_CUDA if device.type == "cuda" else False,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    remove_unused_columns=False,
    report_to=None,  # Disable wandb/tensorboard for now
    seed=RANDOM_SEED,
    data_seed=RANDOM_SEED,
    optim=OPTIMIZER,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=MAX_GRAD_NORM,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,            
    tokenizer=tokenizer
)

print("Starting training...")
print(f"Model: {MODEL_NAME}")
print(f"Model parameters: {model.num_parameters():,}")
print(f"Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Total training steps: {len(tokenized_train_dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")

trainer.train()

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
print(f"Final evaluation results: {eval_results}")
print(f"Final perplexity: {perplexity:.2f}")

print("Training completed successfully!")
