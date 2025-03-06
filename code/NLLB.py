## Evaluation
from utils import direct_translate, back_translate, compute_bleu_chrf
import tqdm
import pandas as pd
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load EHR dataset
medline = pd.read_json('/home/mshahidul/project1/all_tran_data/dataset/Sampled_100_MedlinePlus_eng_spanish_pair.json')
print(f"Total data: {len(medline)}")
# Define NLLB-200 model for translation
model_name = "facebook/nllb-200-3.3B"
# model_name = "facebook/nllb-200-distilled-600M"
# model_name = "facebook/nllb-200-distilled-1.3B"
# model_name = "facebook/nllb-200-1.3B"
model_name2=model_name.split("/")[1]
print(model_name2)
cache_directory = "/data/data_user_alpha/public_models"

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_directory, torch_dtype=torch.float16)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)

translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def translate_nllb_english_to_spanish(text):
    """Translates English text to Spanish using NLLB-200."""
    output = translator(text, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=400)
    return output[0]['translation_text']

output_data = []

for eng, spa in tqdm.tqdm(zip(medline["english"], medline["spanish"])):
    sentence_eng = eng
    sentence_spa = spa  # Reference Spanish translation from dataset
    try:
        # Translation using NLLB
        spa_tran_nllb = translate_nllb_english_to_spanish(sentence_eng)
        # back_tran_nllb = back_translate(spa_tran_nllb) # ENglish convert

        
        reference_text_spa = [sentence_spa]  # For direct translation evaluation

        # Compute BLEU & CHRF scores
        # scores_nllb_back = compute_bleu_chrf(reference_text, back_tran_nllb)
        # scores_direct_back = compute_bleu_chrf(reference_text, back_tran_direct)

        scores_nllb_vs_spa = compute_bleu_chrf(reference_text_spa, spa_tran_nllb)
        # scores_direct_vs_spa = compute_bleu_chrf(reference_text_spa, spa_tran_direct)

        output_data.append({
            "Original_English_sentence": sentence_eng,
            "Original_Spanish_sentence": sentence_spa,
            "spanish_translation_nllb": spa_tran_nllb,
            # "back_translation_nllb": back_tran_nllb,
            # "scores_nllb(bleu and chrf) - Back Translation": scores_nllb_back,
            "scores_nllb(bleu and chrf) - vs medline Spanish": scores_nllb_vs_spa
        })
    except Exception as e:
        print(f"Error: {e}!!!!")
        continue
# Calculate average BLEU score
avg_bleu_score = sum([x['scores_nllb(bleu and chrf) - vs medline Spanish']['bleu_score'] for x in output_data]) / len(output_data)
txt = f"{model_name2}(Medline) --> Average BLEU Score: {avg_bleu_score:.4f}"
print(txt)

# Save results
json_path = f"/home/mshahidul/project1/results_new/medline_{model_name2}_evaluation(spa_ehr).json"
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print(f"Data saved to {json_path}")

