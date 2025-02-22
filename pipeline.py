from transformers import pipeline

def main_pipeline(task, model):
    return pipeline(task=task, model=model)
