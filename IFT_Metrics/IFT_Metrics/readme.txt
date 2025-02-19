eval_metrics_requirements.txt - install packages from here

# to compute BLEURT score - download the model first and change the checkpoint in the eval.py - line 14
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip

# update openapi key if you want to use gpt_4o_mini_as_a_judge.py (note that prompt is specific to summary task)