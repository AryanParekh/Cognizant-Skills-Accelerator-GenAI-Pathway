from datasets import load_dataset

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


from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(
    tf_train_dataset,
    validation_data=tf_test_dataset,
    epochs=3,
)

model.save_pretrained("./fine-tuned-bert-imdb-tf")
tokenizer.save_pretrained("./fine-tuned-bert-imdb-tf")