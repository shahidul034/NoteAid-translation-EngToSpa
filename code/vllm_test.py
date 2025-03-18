
# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# from vllm import LLM, SamplingParams
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)
# prompts = [
#     "Translate the text English to Spanish: Hello, my name is Jemmy.",
#     "Translate the text English to Spanish: The president of the United States is Kane.",
#     "Translate the text English to Spanish: The capital of France is Paris",
#     "Translate the text English to Spanish: The future of AI is bright.",
# ]
# sampling_params = SamplingParams(temperature=1.0, top_p=0.95)
# # llm = LLM(model="unsloth/Qwen2.5-0.5B-Instruct",dtype="half",tensor_parallel_size=4)
# llm = LLM(model="unsloth/Qwen2.5-0.5B-Instruct")
# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


from litgpt import LLM

llm = LLM.load("microsoft/phi-4")
text = llm.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
print(text)
      