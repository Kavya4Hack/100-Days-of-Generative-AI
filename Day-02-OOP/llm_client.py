"""
Day 02 — Object-Oriented Programming
A small provider-independent LLM client architecture.
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Common interface for LLM providers."""

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate(self, prompt):
        """Generate a response from a prompt."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def generate(self, prompt):
        return f"[OpenAI:{self.model}] Response to: {prompt}"


class GeminiClient(LLMClient):
    def generate(self, prompt):
        return f"[Gemini:{self.model}] Response to: {prompt}"


def ask_model(client: LLMClient, prompt: str):
    """Polymorphism: any LLMClient implementation can be used."""
    return client.generate(prompt)


if __name__ == "__main__":
    clients = [
        OpenAIClient("demo-model"),
        GeminiClient("demo-model"),
    ]

    for client in clients:
        print(ask_model(client, "Explain neural networks simply."))
