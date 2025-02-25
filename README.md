feedback.jsonl and translated_inital.jsonl are all used to construct prompt. You can revise them at self-refine/data/tasks/ML/translated_inital.jsonl,self-refine/data/tasks/ML/feedback.jsonl. The test data is self-refine/data/tasks/ML/fed_data.json. 
When testing, please use the command below:
PYTHONPATH=$(pwd) python src/medicalTranslation/run.py 
