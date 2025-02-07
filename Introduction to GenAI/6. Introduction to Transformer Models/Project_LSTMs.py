import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Step 1: Load and Preprocess the Dataset
def load_and_preprocess_data(file_path, sequence_length=50):
    # Load the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Convert to lowercase

    # Create character-level tokenization
    chars = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    # Prepare input and output sequences
    X, y = [], []
    for i in range(0, len(text) - sequence_length):
        seq_in = text[i:i + sequence_length]
        seq_out = text[i + sequence_length]
        X.append([char_to_int[char] for char in seq_in])
        y.append(char_to_int[seq_out])

    # Reshape and normalize input data
    X = np.reshape(X, (len(X), sequence_length))
    y = to_categorical(y, num_classes=len(chars))

    return X, y, chars, char_to_int, int_to_char

# Step 2: Build the LSTM Model
def build_lstm_model(input_shape, output_shape):
    model = Sequential([
        Embedding(input_dim=output_shape, output_dim=50, input_length=input_shape),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(256),
        Dropout(0.2),
        Dense(output_shape, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 3: Train the Model
def train_model(model, X, y, epochs=50, batch_size=128):
    # Save the best model during training
    checkpoint = ModelCheckpoint("LSTM_Model.keras", monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    return model

# Step 4: Generate Text
def generate_text(model, seed_text, char_to_int, int_to_char, sequence_length, num_chars_to_generate=100, temperature=1.0):
    generated_text = seed_text
    for _ in range(num_chars_to_generate):
        # Convert seed text to integer sequence
        seq = [char_to_int[char] for char in seed_text]
        seq = pad_sequences([seq], maxlen=sequence_length, truncating='pre')

        # Predict the next character
        preds = model.predict(seq, verbose=0)[0]
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        preds = np.asarray(preds).astype('float64')
        preds = preds / np.sum(preds)
        next_char = int_to_char[np.argmax(np.random.multinomial(1, preds, 1))]

        # Append the predicted character to the seed text
        generated_text += next_char
        seed_text = seed_text[1:] + next_char

    return generated_text

# Main Execution
if __name__ == "__main__":
    # Parameters
    file_path = "shakespeare.txt" 
    sequence_length = 100
    epochs = 50
    batch_size = 128

    # Step 1: Load and preprocess data
    X, y, chars, char_to_int, int_to_char = load_and_preprocess_data(file_path, sequence_length)

    # Step 2: Build the model
    model = build_lstm_model(X.shape[1], len(chars))
    print(model.summary())

    # Step 3: Train the model
    model = train_model(model, X, y, epochs=epochs, batch_size=batch_size)

    model.save('LSTM_Model.keras')
    model = tf.keras.models.load_model('LSTM_Model.keras')

    # Step 4: Generate text
    seed_text = "shall i compare thee to a summer's day?\n"
    generated_text = generate_text(model, seed_text, char_to_int, int_to_char, sequence_length, num_chars_to_generate=500, temperature=0.5)
    print("\nGenerated Text:\n", generated_text)