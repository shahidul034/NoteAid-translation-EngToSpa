import sys
from typing import Dict, List
from src.utils import Prompt
# from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from llama import Llama
class ResponseGenTaskIterate(Prompt):
    def __init__(self, engine,prompt_examples: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
    #     self.engine = Llama.build(
    #     ckpt_dir="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta-Llama-3-8B/",
    #     tokenizer_path="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta-Llama-3-8B/tokenizer.model",
    #     max_batch_size=1,
    #     max_seq_len=200
    # )
        self.engine = engine
        self.count = 0
        # self.tokenizer = tokenizer
        self.prompt = self.make_prompt(prompt_examples=prompt_examples)

    def make_prompt(self, prompt_examples: str, reduce_window=0) -> str:
        import pandas as pd
        prompt_examples = pd.read_json(prompt_examples, orient="records")
        prompt_examples = prompt_examples[reduce_window:]
        # group on example
        grouped = prompt_examples.groupby("example")
        
        prompt = []
        # sort each group by score
        for _, group in grouped:
            group["numerical_score"] = group["total_score"].apply(lambda x: int(x.split("/")[0].strip()))
            group = group.sort_values("numerical_score")
            prompt.append(self.make_one_iterate_example(group.to_dict("records")))
        
        return self.inter_example_sep.join(prompt) + self.inter_example_sep
        

    def make_one_iterate_example(self, incrementally_improving_examples: List[Dict]):
        """Given a list of examples that are incrementally improving, return a new example.
        """
        
        instr = """We want to iteratively improve the provided translations. To help improve, scores for each translation on desired traits are provided: 1) Clinical usefulness, 2) Clinical Accuracy, 3) Overall Clarity,and 4) Coverage.

"""
        template = """Source clinical note: 

{Originalclinicalnote}

Translation: 

{translation}

Scores:

* Clinical usefulness: {ClinicalUsefulness}
* Clinical Accuracy: {ClinicalAccuracy}
* Overall Clarity: {OverallClarity}
* Coverage: {Coverage}
* Total score: {total_score}

Okay, let's use this feedback to improve the response.

"""     
        prompt = []
        for row in incrementally_improving_examples:
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

        
        prompt = "".join(prompt)
        # prompt = instr + prompt
        # print(f"Prompt: {prompt}")
        return prompt.strip()

    def make_query(self, question: str, reduce_window=0) -> str:
        print(f"question: {question}")
        if reduce_window > 0:
            self.prompt = self.make_prompt(prompt_examples="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/feedback.jsonl", reduce_window=reduce_window)
        return f"{self.prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"

    def _make_input(
        self,
        context: str,
        response: str,
        scores: str,
    ) -> str:
        input_txt = f"""Source clinical note: 

{context}

Translation: 

{response}

Scores:

{scores}

Okay, let's use this feedback to improve the response.

###

Source clinical note: 

{context}
"""

        return input_txt
    def extract_translation(self, text):
        matches = [m.start() for m in re.finditer(r'###', text)]
        if len(matches) < 4:
            return "Not enough sections found."
        section = text[matches[2]:matches[3]]
        translation_match = re.search(r'Translation:\s*(.*)', section, re.DOTALL)
        return translation_match.group(1).strip() if translation_match else "Translation section not found."

    def __call__(
        self,
        responses_to_scores: Dict[str, str],
        reduce_window=0
    ) -> str:
        example_input = self.make_input(
            responses_to_scores=responses_to_scores
        )
        instr = """We want to iteratively improve the provided translations. To help improve, scores for each translation on desired traits are provided: 1) Clinical usefulness, 2) Clinical Accuracy, 3) Overall Clarity,and 4) Coverage.

"""
        transfer_query = instr + self.make_query(example_input, reduce_window=reduce_window)
        self.count += 1
        with open(f"responses_iterate_{self.count}.txt", "w") as f:
            f.write(transfer_query + "\n")
        print(f"Transfer query: {transfer_query}")
        # inputs = self.tokenizer(transfer_query, return_tensors="pt",padding=False).to("cuda")
        # outputs = self.engine.generate(**inputs, temperature=0.7, eos_token_id=self.tokenizer.convert_tokens_to_ids(self.inter_example_sep))
        # input_token_count = inputs['input_ids'].size(1)
        # output_token_count = outputs.size(1) 
        # total_tokens = input_token_count + output_token_count
        # modelresponse = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Response: {modelresponse}")
        modelresponse = self.engine.text_completion(
        [transfer_query],
        max_gen_len=2000,
        temperature=0.6,
        top_p=0.9,
        )
        total_tokens = 0
        modelresponse = modelresponse[0]['generation']
        response = modelresponse.split("Translation:")[1].strip().split("\n")[0].strip()
        print(f"Iteration Response: {response}")
        # print(outputs[0]["generation"])
        return total_tokens, response.strip()


    def make_input(
        self,
        responses_to_scores: Dict[str, str],
    ) -> str:
        input_txt = ""
        for response, (context, scores) in responses_to_scores.items():
            input_txt += self._make_input(
                context=context,
                response=response,
                scores=scores,
            )
        return input_txt


# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-2-7b"
#     tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
#     llama_model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=True, torch_dtype=torch.float16).cuda()
#     obj = ResponseGenTaskIterate(engine=llama_model, tokenizer=tokenizer, prompt_examples="data/prompt/acronym/feedback.v2.jsonl")
#     print(obj.prompt)