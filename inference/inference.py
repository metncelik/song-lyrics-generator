# from transformers import AutoModelForCausalLM, AutoTokenizer
# import re

# tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")
# model = AutoModelForCausalLM.from_pretrained("./gpt2-turkish-song-lyrics/checkpoint-1686")

# input_text = "Uzak "
# inputs = tokenizer.encode(input_text, return_tensors="pt")

# output = model.generate(
#     inputs,
#     max_length=400,
#     num_return_sequences=1,
#     no_repeat_ngram_size=2,
#     top_k=75,
#     top_p=0.95,   
#     temperature=0.7
#     # do_sample=True
# )

# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# generated_text = re.sub(r'(?<!^)([A-Z])', r'\n\1', generated_text)
# print(generated_text)


from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch

tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")
model = AutoModelForCausalLM.from_pretrained("./gpt2-turkish-song-lyrics/checkpoint-1686")

# Add the special tokens used during training
tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    # 'additional_special_tokens': ['[EOL]']
})
model.resize_token_embeddings(len(tokenizer))

def generate_lyrics(prompt, max_length=150, temperature=0.8, top_k=75, top_p=0.95):
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "mps"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=top_k,
        top_p=top_p,   
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
         eos_token_id=tokenizer.convert_tokens_to_ids("test")
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    generated_text = generated_text.replace(' [EOL] ', '\n')
    
    return generated_text

if __name__ == "__main__":
    prompts = [
        "Pankek ",
        "Sevgi ",
        "Hayat "
    ]
    
    for prompt in prompts:
        print(f"\n--- Generating lyrics for prompt: '{prompt}' ---\n")
        lyrics = generate_lyrics(prompt)
        print(lyrics)
        print("\n" + "-" * 50)