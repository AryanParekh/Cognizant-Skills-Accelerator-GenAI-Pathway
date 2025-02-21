from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define the text
text = """The rich man and the poor woman stood side by side, yet their lives were worlds apart. He was wealthy, while she struggled in poverty. The tall boy and the short girl laughed together, but their heights were starkly different. The strong athlete and the weak student both competed, though their abilities varied greatly. Love and hate, joy and sorrow, success and failure—these opposites defined their existence. Yet, in some ways, they were alike: both human, both searching for meaning, and both part of the same world."""

# Tokenize the text
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(tokens))

# Load the pre-trained GPT-2 model
model = TFAutoModel.from_pretrained("gpt2")

# Get embeddings for the tokens
inputs = tokenizer(text, return_tensors="tf")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.numpy()

# Select embeddings for the first 10 tokens
token_embeddings = embeddings[0][:]

print("Shape of embeddings:", token_embeddings.shape)  # Should be (10, embedding_dim)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(token_embeddings)

# Plot the embeddings
plt.figure(figsize=(16, 12))
plt.scatter(reduced_embeddings[:20, 0], reduced_embeddings[:20, 1])
for i, token in enumerate(tokens[:20]):
    plt.annotate(token.removeprefix('Ġ'), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
plt.title("2D Visualization of Token Embeddings (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()