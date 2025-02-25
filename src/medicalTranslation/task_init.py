import pandas as pd
from src.utils import Prompt
from typing import List, Optional, Union
import sys
from prompt_lib.backends import openai_api


class ResponseGenTaskInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str, numexamples=3) -> None:
        super().__init__(
            question_prefix="Original clinical note input: ",
            answer_prefix="Translation: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.setup_prompt_from_examples_file(prompt_examples, numexamples=numexamples)

    def setup_prompt_from_examples_file(self, examples_path: str, numexamples=10) -> str:
        instruction = (
            "Provided an original English clinical note, generate a Spanish translation of this clinical note. Desired traits for the translation are: 1) Clinical usefulness - The translation is useful and can be used in clinical settings, 2) Clinical Accuracy - The translation is clinically accurate, 3) Overall Clarity - The translation is easy to understand, 4) Coverage - The translation covers all important clinical content compared with the original note, and 5) Fluent. Response should begin with - Translation:\n\n"
        )

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for i, row in examples_df.iterrows():
            if i >= numexamples:
                break
            prompt.append(self._build_query_from_example(row["Original clinical note"], row["translation"]))

        self.prompt = instruction + self.inter_example_sep.join(prompt) + self.inter_example_sep

    def _build_query_from_example(self, sourceNote: Union[str, List[str]], translation: Optional[str]=None) -> str:
        # history = history.replace('System: ', '').replace('User: ', '')

        TEMPLATE = """Original clinical note: 

{sourceNote}

Translation: {translation}"""
            
        query = TEMPLATE.format(sourceNote=sourceNote, translation=translation)
        return query

    def make_query(self, source: str) -> str:
        query = f"{self.prompt}{self.question_prefix}\n\n{source}{self.intra_example_sep}"
        return query

    def __call__(self, source: str) -> str:
        generation_query = self.make_query(source)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=800,
            stop_token="###",
            temperature=0.7,
        )

        generated_response = openai_api.OpenaiAPIWrapper.get_first_response(output)

        generated_response = generated_response.split(self.answer_prefix)[1].replace("#", "").strip()


        return output, generated_response.strip()
