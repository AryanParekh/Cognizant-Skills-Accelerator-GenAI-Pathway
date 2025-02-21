from datasets import load_dataset
import tensorflow as tf

# Load IMDb dataset
dataset = load_dataset("imdb")

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols=["label"],
    shuffle=True,
    batch_size=16,
)

tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols=["label"],
    shuffle=False,
    batch_size=16,
)

from transformers import TFBertForSequenceClassification, BertTokenizer

model_path = "./fine-tuned-bert-imdb-tf"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the model
model = TFBertForSequenceClassification.from_pretrained(model_path)

predictions = model.predict(tf_test_dataset)
preds = tf.argmax(predictions.logits, axis=-1)

from sklearn.metrics import accuracy_score

labels = tokenized_datasets["test"]["label"]
accuracy = accuracy_score(labels, preds)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import f1_score

f1 = f1_score(labels, preds)
print(f"F1-Score: {f1}")