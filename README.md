feedback.jsonl and translated_inital.jsonl are all used to construct prompt. You can revise them at self-refine/translated_inital.jsonl,self-refine/feedback.jsonl. The test data is self-refine/fed_data.json. 

First, cd ./self-refine
When testing on model GPT, please use the command below:
```bash
PYTHONPATH=$(pwd) python src/medicalTranslation/run.py 
```
When testing on model Llama, please use the command below:
```bash
PYTHONPATH=$(pwd) python src/medicalTranslation/run_Llama.py 
```

After getting the result, then you can use self-refine/ConstructData.ipynb to extract translation.
