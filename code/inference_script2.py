import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# input_text = "Translate the input into English to Spanish: I love programming."
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""
model_id  = "projecte-aina/aguila-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
generation = generator(
    input_text,
    do_sample=True,
    top_k=10,
    eos_token_id=tokenizer.eos_token_id,
)

print(f"Result: {generation[0]['generated_text']}")
