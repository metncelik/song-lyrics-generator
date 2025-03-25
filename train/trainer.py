import re
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset
sys.path.append(str(Path(__file__).parent.parent))
from database.client import DatabaseClient

# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.distributed.xla_multiprocessing as xmp

db_client = DatabaseClient()

model_name = "ByteWave/gpt2-turkish-uncased"

raw_lyrics_data = db_client.get_lyrics()

def process_data(lyrics_item):
    lyrics = lyrics_item[0]
    song_title = lyrics_item[1]
    artist_name = lyrics_item[2]

    lyrics = "<|startofsong|><|startofline|>" + \
        lyrics + "<|endofline|><|endofsong|>"
    lyrics = re.sub(r'\n', '<|endofline|><|startofline|>', lyrics)
    
    unwanted = [
        "\"",
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "-",
        "_",
        "=",
        "+",
        "*",
        "/",
        ")",
        "(",
        "[",
        "]",
        "{",
        "}",
        ",",
        "'",
        "`",
        "~",
        "|",
        "@"
    ]
    
    for unwanted in unwanted:
        lyrics = lyrics.replace(unwanted, "")
    
    lyrics = lyrics.lower()
    

    return {
        "lyrics": lyrics,
        "song_title": song_title,
        "artist_name": artist_name
    }


processed_lyrics = [process_data(item) for item in raw_lyrics_data]

lyrics_dataset = Dataset.from_list(processed_lyrics)

train_test_split = lyrics_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    'additional_special_tokens': [
        '<|startofline|>', '<|endofline|>',
        '<|startofsong|>', '<|endofsong|>'
    ]
})

tokenizer.save_pretrained("./tokenizer")

def tokenize_function(examples):
    return tokenizer(examples["lyrics"], padding=True, truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(
    ["lyrics", "song_title", "artist_name"])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(
    ["lyrics", "song_title", "artist_name"])

tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="gpt2-turkish-song-lyrics",
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    warmup_steps=500,
    fp16=True,
    gradient_accumulation_steps=8,
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

eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
print(f"Evaluation results: {eval_results}, Perplexity: {perplexity}")
