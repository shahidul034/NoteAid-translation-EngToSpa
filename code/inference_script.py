from utils import compute_bleu_chrf
import json
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from unsloth import FastLanguageModel
file_path = "/home/mshahidul/project1/all_tran_data/dataset/Sampled_100_MedlinePlus_eng_spanish_pair.json"
with open(file_path, 'r', encoding='utf-8') as json_file:
    original_file = json.load(json_file)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""
max_seq_length=2048
model_name = "unsloth/llama-3-8b-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

total_score=[]
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

def inference(ques):
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Translate the input into English to Spanish:", # instruction
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


for line in tqdm.tqdm(original_file):
    hypothesis_text = inference(line['english'])
    reference_text = line['spanish']
    score=compute_bleu_chrf(reference_text, hypothesis_text)  
    total_score.append({
        "original_english": line['english'],
        "original_spanish": line['spanish'],
        "translated_spanish": hypothesis_text,
        "bleu_score": score
    })

tt=model_name.split("/")[1]
avg_bleu_score = sum([x['bleu_score']['bleu_score'] for x in total_score]) / len(total_score)

print(f"{tt} without finetune --> Average BLEU Score: {avg_bleu_score:.4f}")

# with open(f"/home/mshahidul/project1/results/{tt}_without_finetune.json", 'w', encoding='utf-8') as json_file:
#     json.dump(total_score, json_file, ensure_ascii=False, indent=4)