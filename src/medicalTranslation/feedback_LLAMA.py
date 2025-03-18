import pandas as pd
# from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from src.utils import Prompt
from llama import Llama
import re
class ResponseGenFeedback(Prompt):
    def __init__(self,engine,prompt_examples: str) -> None:
        super().__init__(
            question_prefix="Source Original Note",
            answer_prefix="Translation:",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
    #     self.engine =Llama.build(
    #     ckpt_dir="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta-Llama-3-8B/",
    #     tokenizer_path="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta-Llama-3-8B/tokenizer.model",
    #     max_batch_size=1,
    #     max_seq_len=max_tokens
    # )
        
        
        self.engine = engine
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        template = """Source clinical note:

{Originalclinicalnote}
        
Translation: 

{translation}

Scores:

* Clinical usefulness: {ClinicalUsefulness}
* Clinical Accuracy: {ClinicalAccuracy}
* Overall Clarity: {OverallClarity}
* Coverage: {Coverage}
* Total score: {total_score}"""
        examples_df = pd.read_json(examples_path, orient="records")
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(
                template.format(
                    Originalclinicalnote=row['Original clinical note'],
                    translation=row["translation"],
                    ClinicalUsefulness=row["Clinical usefulness"],
                    ClinicalAccuracy=row["Clinical Accuracy"],
                    OverallClarity=row["Overall Clarity"],
                    Coverage=row["Coverage"],
                    # Fluent=row["Fluent"],
                    total_score=row["total_score"],
                )
            )

        instruction = """We want to iteratively improve the provided translation. To help improve, scores for each translation on desired traits are provided: 1) Clinical usefulness, 2) Clinical Accuracy, 3) Overall Clarity, and 4) Coverage.

Here are some examples of this scoring rubric:

"""
        self.prompt = instruction + self.inter_example_sep.join(prompt)+ self.inter_example_sep
        # self.prompt = self.inter_example_sep.join(prompt) 
        # print(f"Prompt: {self.prompt}")
    def extract_translation(self, text):
        matches = [m.start() for m in re.finditer(r'###', text)]
        if len(matches) < 4:
            return "Not enough sections found."
        section = text[matches[2]:matches[3]]
        translation_match = re.search(r'Scores:\s*(.*)', section, re.DOTALL)
        return translation_match.group(1).strip() if translation_match else "Translation section not found."
    def get_score_details(self,text):
        match = re.search(r'((?:.*\n){5}.*)Total score:', text)
        return match.group(1).strip() if match else None

    def __call__(self, context: str, response: str):
        prompt = self.get_prompt_with_question(context=context, response=response)
        # print(f"Feedback Prompt: {prompt}")
        # inputs = self.tokenizer(prompt, return_tensors="pt",padding=False).to("cuda")
        # outputs = self.engine.generate(**inputs, temperature=0.7, eos_token_id=self.tokenizer.convert_tokens_to_ids("###"),max_length=800)
        # input_token_count = inputs['input_ids'].size(1)
        # output_token_count = outputs.size(1) 
        # total_tokens = input_token_count + output_token_count
        # generated_feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        results = self.engine.text_completion(
        [prompt],
        max_gen_len=800,
        temperature=0.6,
        top_p=0.9,
        )
        generated_feedback = results[0]['generation']
        # print(f"Generated Feedback: {generated_feedback}")
        generated_feedback = generated_feedback.split("Scores:")[1].strip()
        generated_feedback = generated_feedback.split("#")[0].strip()
        # print(f"Generated Feedback: {generated_feedback}")
        # print("-----------------------")
        # generated_feedback = self.extract_translation(generated_feedback)
        # print(f"Generated Feedback2: {generated_feedback}")
        return 0, generated_feedback

    def get_prompt_with_question(self, context: str, response: str):
        question = self.make_query(context=context, response=response)
        # print(f"prompt:{self.prompt}")
        # print(f"question:{question}")
        return f"""{self.prompt}{question}\n"""

    def make_query(self, context: str, response: str):
        question = f"""Source clinical note: 

{context}

Translation: 

{response}"""
        return question