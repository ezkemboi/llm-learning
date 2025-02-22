from pipeline import main_pipeline

generator = main_pipeline("text-generation", "distilgpt2")

prompt = 'Momi Foundation from Kenya is'

output = generator(prompt, max_length=100, pad_token_id=generator.tokenizer.eos_token_id, truncation=True)

print(output[0]['generated_text'])
