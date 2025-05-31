from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import process_lyrics
import sys

class Model:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model.to(self.device)
        # self.add_special_tokens()

    def add_special_tokens(self):
        self.tokenizer.add_special_tokens({
            'pad_token': '<|pad|>',
            # 'additional_special_tokens': [
            #     '<|startofline|>', 
            #     '<|endofline|>',
            #     '<|startofsong|>', '<|endofsong|>'
            # ]
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

    def generate_lyrics(self, prompt, max_length=150, temperature=0.8, top_k=75, top_p=0.95):
        prompt = process_lyrics(prompt)
        
        if not prompt.strip():
            prompt = "."
        
        inputs = self.tokenizer.encode_plus(
            prompt,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=top_k,
            top_p=top_p,   
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        lines = generated_text.split(".")
        lines = [line.capitalize() for line in lines] 
        generated_text = "\n".join(lines)
        if generated_text.endswith("."):
            generated_text = generated_text[:-1]
        if generated_text.startswith("."):
            generated_text = generated_text[1:]
        generated_text = generated_text.replace(".", "\n")
        
        return generated_text

if __name__ == "__main__":
    if len(sys.argv) > 1 and "-generate" in sys.argv:
        prompt = sys.argv[2] if len(sys.argv) > 2 else "."
        model_path = sys.argv[3] if len(sys.argv) > 3 else "metncelik/tr-lyrics-generator-cosmos-gpt2-large"
        model = Model(model_path)
        lyrics = model.generate_lyrics(prompt)
        print(lyrics)
        print(f"Model: {model_path}")
    else:
        print("Usage: python inference.py -generate <prompt> <model_path>")