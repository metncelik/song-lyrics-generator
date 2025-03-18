from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "ytu-ce-cosmos/turkish-gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "türkiye bir "
inputs = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=75,
    top_p=0.95,   
    temperature=0.8,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)