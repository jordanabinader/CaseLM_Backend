from openai import OpenAI

# Initialize the client with your API key
client = OpenAI()  # Reads from OPENAI_API_KEY environment variable
# Or explicitly: client = OpenAI(api_key="your-api-key")

try:
    response = client.chat.completions.create(
        model="gpt-4",  # Note: "gpt-40" was incorrect
        messages=[
            {"role": "user", "content": "Hello, world!"}
        ],
        temperature=0.7  # Optional: controls response randomness
    )
    # Access the response content
    print(response.choices[0].message.content)
except Exception as e:
    print(f"API Error: {e}")