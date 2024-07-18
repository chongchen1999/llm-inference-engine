import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load model and tokenizer
model_path = "/home/tourist/AI-HPC Projects/llm_inference_engine/model_zoo/llama-2-7b-hf-16"
print("1")

model = LlamaForCausalLM.from_pretrained(model_path).to('cuda')
print("2")

tokenizer = AutoTokenizer.from_pretrained(model_path)
print("3")

# Define prompt
prompt = "Hey, are you conscious? Can you talk to me?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
print(inputs)
print("4")

# Generate response
generate_ids = model.generate(inputs.input_ids, max_length=30)

# Decode generated token IDs back into a string
output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

# Print the output
print(output)
