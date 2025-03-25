from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "ByteWave/gpt2-turkish-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    # 'additional_special_tokens': [
    #     '<|startofline|>', '<|endofline|>',
    #     '<|startofsong|>', '<|endofsong|>'
    # ]
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
        bos_token_id=tokenizer.convert_tokens_to_ids("<|startofsong|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endofsong|>")
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    return generated_text

if __name__ == "__main__":
    prompts = [
        "futbol ",
        "tekvando ",
        "aziz yıldırım "
    ]
    
    for prompt in prompts:
        print(f"\n--- Generating lyrics for prompt: '{prompt}' ---\n")
        lyrics = generate_lyrics(prompt)
        print(lyrics)
        print("\n" + "-" * 50)