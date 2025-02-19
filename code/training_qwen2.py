import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
model_name = "unsloth/Qwen2.5-32B-bnb-4bit"
with open("/home/mshahidul/project1/all_tran_data/medline_data_for_finetune.json") as f:
    data = json.load(f)

# Convert to the required format
converted_dataset = []

for entry in data:
    conversation={}
    conversation['conversations']= (
        {"from": "human", "value": f"Translate the sentence into English to Spanish: {entry['english']}"},  
        {"from": "gpt", "value": entry["spanish"]}
    )
    converted_dataset.append(conversation)
from datasets import Dataset
dataset = Dataset.from_list(converted_dataset)
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)
trainer_stats = trainer.train()
import re
def extract_translation(text):
    match = re.search(r"Spanish: (.*?)<\|im_end\|>", text)
    
    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        text=text.split("\n")
        text=text[len(text)-1]
        text=text.split("<|im_end|>")[0]
        return text
    
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

def inference(text):
    messages = [
            {"role": "user", "content": f"Translate the input into English to Spanish: \n\nEnglish: {text}\n\nSpanish:"},]
    inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 1.0, min_p = 1.0)
    temp= tokenizer.batch_decode(outputs)[0]
    return extract_translation(temp)

file_path = "/home/mshahidul/project1/all_tran_data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
with open(file_path, 'r', encoding='utf-8') as json_file:
    original_file = json.load(json_file)
import tqdm
from utils import compute_bleu_chrf
total_score=[]
for line in tqdm.tqdm(original_file):
    try:
        hypothesis_text = inference(line['english'])
        reference_text = line['spanish']
        score=compute_bleu_chrf(reference_text, hypothesis_text)  
        total_score.append({
            "original_english": line['english'],
            "original_spanish": line['spanish'],
            "translated_spanish": hypothesis_text,
            "bleu_score": score
        })
    except Exception as e:
        print(e)
        continue

model_name=model_name.split("/")[1]
avg_bleu_score = sum([x['bleu_score']['bleu_score'] for x in total_score]) / len(total_score)

print(f"{model_name} with finetune --> Average BLEU Score: {avg_bleu_score:.4f}")

with open(f"/home/mshahidul/project1/results/{model_name}_finetuned.json", 'w', encoding='utf-8') as json_file:
    json.dump(total_score, json_file, ensure_ascii=False, indent=4)