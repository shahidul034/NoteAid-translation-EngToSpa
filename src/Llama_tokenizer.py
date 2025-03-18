from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
model_name = "./llama3-demo/llama3/Llama-3.2-1B"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name) # Replace with your tokenizer name
model = LlamaForCausalLM.from_pretrained(model_name) # Replace with your model name

# Add special tokens to the tokenizer
special_tokens_dict = {"additional_special_tokens": ["###","\n\n###\n\n","\n\n"]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# Resize token embeddings with mean_resizing option
model.resize_token_embeddings(len(tokenizer) + num_added_toks, mean_resizing=True)

# Set pad token ID
tokenizer.pad_token = tokenizer.eos_token

# Example usage of the model.generate with stop tokens
input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
stop_token = "###"  # Define the stop token

# Generate text
output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, eos_token_id=tokenizer.convert_tokens_to_ids(stop_token), pad_token_id=tokenizer.eos_token_id)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)