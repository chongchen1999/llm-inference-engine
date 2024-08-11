import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load model and tokenizer
model_path = "/home/tourist/AI-HPC Projects/llm_inference_engine/model_zoo/Llama-2-7b-chat-hf-4bit"

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_path)

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
