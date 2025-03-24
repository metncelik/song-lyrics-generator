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

model_name = "redrussianarmy/gpt2-turkish-cased"

raw_lyrics_data = db_client.get_lyrics(25000)


def process_data(lyrics_item):
    lyrics = lyrics_item[0]
    song_title = lyrics_item[1]
    artist_name = lyrics_item[2]

    # <|startofsong|> - start of song, <|startofline|> - start of line,
    # <|endofline|> - end of line - <|endofsong|> - end of song

    # lyrics = re.sub("\"", "", lyrics)
    # lyrics = "<|startofsong|><|startofline|>" + lyrics
    # lyrics = re.sub(r'\n', '<|endofline|><|startofline|>', lyrics)
    # lyrics = lyrics + "<|endofline|><|endofsong|>"
    # lyrics = re.sub(r'\s+', ' ', lyrics)

    lyrics = "<|startofsong|><|startofline|>" + \
        lyrics + "<|endofline|><|endofsong|>"
    lyrics = re.sub(r'\n', '<|endofline|><|startofline|>', lyrics)

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

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    'additional_special_tokens': [
        '<|startofline|>', '<|endofline|>',
        '<|startofsong|>', '<|endofsong|>'
    ]
})

print(tokenizer.special_tokens_map)

tokenizer.save_pretrained("./tokenizer")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(
    ["text", "song_title", "artist_name"])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(
    ["text", "song_title", "artist_name"])

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
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=1000,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    warmup_steps=200,
    # tpu_num_cores=8,  
    gradient_accumulation_steps=4,
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
