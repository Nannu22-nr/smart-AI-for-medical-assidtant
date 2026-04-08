import google.generativeai as genai

# 🔑 Replace with your API key
genai.configure(api_key="AIzaSyC8JqTyxK127HQU8SyO_admvbv8RVabeCg")

# Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# Send a test prompt
response = model.generate_content("Say hello")

print(response.text)