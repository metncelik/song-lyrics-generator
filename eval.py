import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from datasets import Dataset
sys.path.append(str(Path(__file__).parent.parent))
from database.client import DatabaseClient
from utils import process_data

# Get model path from command line arguments
if len(sys.argv) < 2:
    print("Usage: python eval.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]

db_client = DatabaseClient()
raw_songs_data = db_client.get_songs()


processed_songs = [process_data(item) for item in raw_songs_data]

songs_dataset = Dataset.from_list(processed_songs)

train_test_split = songs_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    # 'additional_special_tokens': [
    #     '<|startofline|>', '<|endofline|>',
    #     '<|startofsong|>', '<|endofsong|>'
    # ]
})

tokenizer.save_pretrained("./tokenizer")

def tokenize(examples):
    return tokenizer(examples["lyrics"], padding=True, truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(
    ["lyrics", "song_title", "artist_name"])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(
    ["lyrics", "song_title", "artist_name"])

tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

model = AutoModelForCausalLM.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

model.eval()

def calculate_perplexity(model, dataset, tokenizer, batch_size=8):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        print(f"len(dataset) {len(dataset)}")
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            print(f"loss {loss}")
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    
    print(f"total_loss {total_loss}")
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

perplexity = calculate_perplexity(model, tokenized_eval_dataset.select(range(500)), tokenizer)
print(f"Perplexity on evaluation dataset: {perplexity:.2f}")

