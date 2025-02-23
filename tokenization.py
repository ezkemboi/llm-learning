from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


train_data = load_dataset('imdb', split="train")
train_data = train_data.shard(num_shards=4, index=0)

test_data = load_dataset('imdb', split="test")
test_data = test_data.shard(num_shards=4, index=0)

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

tokenized_training_data = tokenizer(train_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64)
tokenized_test_data = tokenizer(test_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64)

print(tokenized_training_data)

# Sub-word tokenization and only accept DataSet
# Tokenizing row by row
def tokenize_data(text_data):
    return tokenizer(text_data["text"], return_tensors='pt', padding=True, truncation=True, max_length=64)

# tokenize in batches
tokenized_in_batches = train_data.map(tokenize_data, batched=True)
# tokenize row by row
tokenized_by_row = train_data.map(tokenize_data, batched=False)