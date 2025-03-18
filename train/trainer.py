import re
import sys
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset
sys.path.append(str(Path(__file__).parent.parent))
from database.client import DatabaseClient

db_client = DatabaseClient()

raw_lyrics_data = db_client.get_lyrics(12)

def process_data(lyrics_item):
    lyrics = lyrics_item[0] 
    song_title = lyrics_item[1]  
    artist_name = lyrics_item[2] 
    
    lyrics = re.sub("\"", "", lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics) 
    
    return {
        "text": lyrics,
        "song_title": song_title,
        "artist_name": artist_name
    }

processed_lyrics = [process_data(item) for item in raw_lyrics_data]

lyrics_dataset = Dataset.from_list(processed_lyrics)

train_test_split = lyrics_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained("redrussianarmy/gpt2-turkish-cased")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

training_args = TrainingArguments(
    output_dir="gpt2-turkish-song-lyrics",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Run evaluation
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")