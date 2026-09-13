# import requests
# # GET method
# response = requests.get(
#     "https://api.example.com/users/123"
# )
# print(response.status_code)
# print(response.json())

# Post Method
# import requests
# data = {
#     "name" : "Kavya",
#     "role" : "AI/ML Engineer",
#     "age" : 10
# }
# response = requests.post(
#     "https://api.example.com/users",
#     json = data
# )
# print(response.json())

# import os
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")

# print(api_key)

import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

url = "https://techniver.com/google-gemini-api-url/"

headers = {
    "Content-Type" : "application/json"
}
payload = {
    "contents" : [
        {
            "parts" : [
                {
                    "text" : "Explain machine learining in simple words."
                }
            ]
        }
    ]
}
response = requests.post(
    url,
    headers=headers,
    json=payload
)

response.raise_for_status()
data = response.json()
print(data)

# if response.status_code == 200:
#     data = response.json()
#     print(data)
# else:
#     print("Request failed")
#     print(response.status_code)
#     print(response.text)

"""
    Build a Reusable AI Client
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class GeminiClent():
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')

    def generate(self, prompt):
        # Build request
        # Send request
        # Parse response
        # Return generated text
        pass
client = GeminiClent()
response = client.generate(
    "Explain neural netwirk."
)
print(response)