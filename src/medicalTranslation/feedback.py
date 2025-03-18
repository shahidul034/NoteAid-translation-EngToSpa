import pandas as pd
from prompt_lib.backends import openai_api

from src.utils import Prompt


class ResponseGenFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, max_tokens: int = 400) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.max_tokens = max_tokens
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

        instruction = """We want to iteratively improve the provided translation. To help improve, scores for translation are provided: 1) Clinical usefulness, 2) Clinical Accuracy, 3) Overall Clarity, 4) Coverage, and 5) Fluent.

Here are some examples of this scoring rubric:

"""
        # self.prompt = instruction + self.inter_example_sep.join(prompt)
        self.prompt = instruction + self.inter_example_sep.join(prompt) + self.inter_example_sep
        
        
    
    def __call__(self, context: str, response: str):
        prompt = self.get_prompt_with_question(context=context, response=response)
        # print(f"FeedbackPrompt: {prompt}")
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=prompt,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="###",
            temperature=0.7,
        )
        
        
        generated_feedback = openai_api.OpenaiAPIWrapper.get_first_response(output)
        # print(f"Generated Feedback: {generated_feedback}")
        generated_feedback = generated_feedback.split("Scores:")[1].strip()
        generated_feedback = generated_feedback.split("#")[0].strip()
        # print(f"Generated Feedback: {generated_feedback}")
        return output, generated_feedback

    def get_prompt_with_question(self, context: str, response: str):
        # context = context.replace('System: ', '').replace('User: ', '')
        # print("PROMPT", self.prompt)
        question = self.make_query(context=context, response=response)
        return f"""{self.prompt}{question}\n\n"""

    def make_query(self, context: str, response: str):
        question = f"""Source clinical note: 

{context}

Translation: 

{response}"""
        return question
