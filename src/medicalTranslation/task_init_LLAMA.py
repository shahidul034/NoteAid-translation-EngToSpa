import pandas as pd
from src.utils import Prompt
from typing import List, Optional, Union
import sys
# from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
# from llama import Llama
from llama import Llama
import re
class ResponseGenTaskInit(Prompt):
    def __init__(self, engine,prompt_examples: str,  numexamples=3) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="Translation: ",
            intra_example_sep="\n",
            inter_example_sep="\n",
        )
        # self.engine = Llama.build(
        #     ckpt_dir="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta-Llama-3-8B/",
        #     tokenizer_path="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta-Llama-3-8B/tokenizer.model",
        #     max_batch_size=1,
        #     max_seq_len=800,
        #     # stop_token="###",
        #     )
        self.engine = engine
        self.setup_prompt_from_examples_file(prompt_examples, numexamples=numexamples)

    def setup_prompt_from_examples_file(self, examples_path: str, numexamples=10) -> str:
        instruction = (
            "Translate the English clinical note to Spanish. Desired traits for the translation are: 1) Clinical usefulness - The translation is useful and can be used in clinical settings, 2) Clinical Accuracy - The translation is clinically accurate, 3) Overall Clarity - The translation is easy to understand, and 4) Coverage - The translation covers all important clinical content compared with the original note.\n\n"
        )

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for i, row in examples_df.iterrows():
            if i >= numexamples:
                break
            prompt.append(self._build_query_from_example(row["Original clinical note"], row["translation"]))

        self.prompt = instruction + self.inter_example_sep.join(prompt) + self.inter_example_sep

    def _build_query_from_example(self, sourceNote: Union[str, List[str]], translation: Optional[str]=None) -> str:
        TEMPLATE = """{sourceNote}={translation}"""
        query = TEMPLATE.format(sourceNote=sourceNote, translation=translation)
        return query

    def make_query(self, source: str) -> str:
        query = f"{self.prompt}{source}={self.intra_example_sep}"
        return query
    

   
    def extract_translation(self, text):
        matches = [m.start() for m in re.finditer(r'###', text)]
        if len(matches) < 4:
            return "Not enough sections found."
        section = text[matches[2]:matches[3]]
        translation_match = re.search(r'Translation:\s*(.*)', section, re.DOTALL)
        return translation_match.group(1).strip() if translation_match else "Translation section not found."

    def get_translation_content(self,text):
        match = re.search(r'(.*?)(\n|$)', text)
        return match.group(1).strip() if match else None

    def __call__(self, source: str) -> str:
        generation_query = self.make_query(source)
        print(f"source:{generation_query}")
        # # generated_response = generated_response.split(self.answer_prefix)[1].replace("#", "").strip()
        # generated_response = self.extract_translation(generated_response)
        results = self.engine.text_completion(
        [generation_query],
        max_gen_len=800,
        temperature=0.6,
        top_p=0.9,
        )
        generated_response = self.get_translation_content(results[0]['generation']).strip()
        print(f"Generated initial response: {generated_response}")
        return 0,generated_response.strip()