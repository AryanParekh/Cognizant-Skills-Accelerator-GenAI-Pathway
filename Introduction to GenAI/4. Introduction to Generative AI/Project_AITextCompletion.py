import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    # model="meta-llama/Meta-Llama-3-8B-Instruct",
    token="_" # Removed for security purpose
)

print("Enter text to have conversation with AI. Enter 'exit' or 'quit' to end the conversation. Have fun!")
print()
messages = []
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Conversation ends. Goodbye!")
        break
    if not user_input.strip():
        print("Please enter a valid text.")
        continue
    messages.append({'role':'user','content':user_input})
    try:
        result = client.chat_completion(messages, max_tokens=150, seed=42)
        print(f"AI: {result.choices[0].message.content}")
        messages.append({'role':'assistant','content':result.choices[0].message.content})
    except Exception as e:
        print(f"An error occurred: {e}")
        break