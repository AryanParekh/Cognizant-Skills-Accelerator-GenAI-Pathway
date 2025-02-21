import streamlit as st
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModel.from_pretrained("bert-base-uncased")

# Define FAQs
faqs = {
    "How do I reset my password?": "To reset your password, go to the login page and click 'Forgot Password.' Follow the instructions sent to your email.",
    "What are your business hours?": "Our business hours are Monday to Friday, 9 AM to 5 PM.",
    "How can I contact support?": "You can contact support by emailing support@example.com or calling +1-800-123-4567.",
    "Where can I find my order history?": "You can find your order history by logging into your account and navigating to the 'Orders' section.",
    "Do you offer refunds?": "Yes, we offer refunds within 30 days of purchase. Please contact support to initiate a refund.",
}

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    return np.mean(outputs.last_hidden_state.numpy(), axis=1)

# Precompute FAQ embeddings
faq_embeddings = {question: get_embedding(question) for question in faqs.keys()}

# Streamlit app
st.title("FAQ Assistant")
user_query = st.text_input("Ask a question:")

if user_query:
    # Get embedding for user query
    query_embedding = get_embedding(user_query)

    # Compute similarity with FAQs
    similarities = {
        question: cosine_similarity(query_embedding, faq_embedding)[0][0]
        for question, faq_embedding in faq_embeddings.items()
    }

    # Find the most relevant FAQ
    most_relevant_question = max(similarities, key=similarities.get)
    answer = faqs[most_relevant_question]

    # Display the answer
    # st.write(f"**Question:** {most_relevant_question}")
    st.write(f"**Answer:** {answer}")