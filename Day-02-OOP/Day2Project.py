from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass

class GeminiClient(LLMClient):
    def generate(self, prompt):
        return f"Gemin response: {prompt}"

class OpenAIClient(LLMClient):
    def generate(self, prompt):
        return f"OpenAI response: {prompt}"

gemini = GeminiClient()
openai = OpenAIClient()

print(gemini.generate("RAG"))
print(openai.generate("RAG"))

class MockLLMClient(LLMClient):
    def generate(self, prompt):
        return "This is a test response"

clients = [
    GeminiClient(),
    OpenAIClient(),
    MockLLMClient()
]
for client in clients:
    print(client.generate("Explain AI agents."))