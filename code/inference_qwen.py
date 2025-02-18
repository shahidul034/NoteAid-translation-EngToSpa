from utils import compute_bleu_chrf
import json
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from unsloth import FastLanguageModel
file_path = "/home/mshahidul/project1/all_tran_data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
with open(file_path, 'r', encoding='utf-8') as json_file:
    original_file = json.load(json_file)


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Translate the input into English to Spanish: \n\nEnglish: What is\n\nSpanish:"},]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,temperature = 1.5, min_p = 0.1)
print(tokenizer.batch_decode(outputs))