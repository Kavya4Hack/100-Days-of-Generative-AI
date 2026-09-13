import os, requests
from dotenv import load_dotenv

load_dotenv()

class GeminiClient():
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
    def generate(self, prompt):
        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/gemini-2.5-flash:generateContent"
        )
        headers = {
            'content-type' : 'application/json',
            'x-goog-api-key' : self.api_key
        }
        payload = {
            'contents' : [
                {
                    'parts' : [
                        {
                            'text' : prompt
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
        return data
gemini = GeminiClient()

response = gemini.generate(
    "Explain transformer in simple words."
)
print(response)