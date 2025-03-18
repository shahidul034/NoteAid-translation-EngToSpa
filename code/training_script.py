import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import json
model_name = "unsloth/llama-3-8b-Instruct"

with open("/home/mshahidul/project1/all_tran_data/dataset/medline_data_for_finetune.json") as f:
    data = json.load(f)
from datasets import Dataset
dataset = Dataset.from_list(data)
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False,
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
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = "Translate the input into English to Spanish:"
    inputs       = examples["english"]
    outputs      = examples["spanish"]
    texts = []
    for input, output in zip( inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instructions, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = dataset.map(formatting_prompts_func, batched = True,)
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 20,
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
trainer_stats = trainer.train()
# model.save_pretrained("/home/mshahidul/project1/model/unsloth/DeepSeek-R1-Distill-Qwen-14B")  # Local saving
# tokenizer.save_pretrained("/home/mshahidul/project1/model/unsloth/DeepSeek-R1-Distill-Qwen-14B")

from utils import compute_bleu_chrf
path = "/home/mshahidul/project1/all_tran_data/dataset/Sampled_100_MedlinePlus_eng_spanish_pair.json"
with open(path) as f:
    data = json.load(f)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""
max_seq_length=2048
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
ans_cal=[]
# alpaca_prompt = Copied from above
results_file_path = "/home/mshahidul/project1/results_new/Medline/medlineplus_gpt4_mini_COD_back_translation.json"
with open(results_file_path, 'r', encoding='utf-8') as json_file:
    results_data = json.load(json_file)
sentence_to_prompt = {item['Original_English_sentence']: item['COD_prompt'] for item in results_data}
def find_cod_prompt(english_sentence):
    return sentence_to_prompt.get(english_sentence, "Prompt not found")

def inference(ques):
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            f"{find_cod_prompt(ques)}\nTranslate the input into English to Spanish using above context:", # instruction
            ques, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    ans=tokenizer.batch_decode(outputs)
    start_marker = '### Response:\n'
    end_marker = '<|end_of_text|>'

    # Find the start and end positions
    start_index = ans[0].find(start_marker) + len(start_marker)
    end_index = ans[0].find(end_marker)
    response = ans[0][start_index:end_index].strip()
    return response 

# Extract the response
import tqdm
for x in tqdm.tqdm(data):
    try:
        hyp=inference(x['english'])
        ref=x['spanish']
        score=compute_bleu_chrf(ref, hyp)
        ans_cal.append({
            "original_eng":x['english'],
            "original_spa":x['spanish'],
            "tran_spa":hyp,
            "bleu_score":score['bleu_score'],
            "chrf_score":score['chrF++']
        })
    except:
        pass
tt=model_name.split("/")[1]
avg=0
avg_chrf=0
for x in ans_cal:
    avg+=x['bleu_score']
    avg_chrf+=x['chrf_score']
avg/=len(ans_cal)
avg_chrf/=len(ans_cal)
print(f"{model_name} with finetune (COD): {avg}")
# import json
# with open(f"/home/mshahidul/project1/results/finetuned_{tt}.json", 'w') as f:
#     json.dump(ans_cal, f)   