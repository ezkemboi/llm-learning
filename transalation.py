from pipeline import main_pipeline

translator = main_pipeline('translation_es_to_en', "Helsinki-NLP/opus-mt-es-en")

spanish_text = "Este curso sobre LLMs se está poniendo muy interesante"

transalations = translator(spanish_text, clean_up_tokenization_spaces=True)

print(transalations[0]["translation_text"])