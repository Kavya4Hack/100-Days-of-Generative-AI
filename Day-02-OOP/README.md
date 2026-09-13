# Day 02 — Object-Oriented Programming 🏗️

## 🎯 Objective
Learn the OOP principles needed to design maintainable and extensible AI applications.

## 📚 Concepts Covered
- Classes and objects
- Constructors
- Inheritance
- Polymorphism
- Encapsulation
- Abstraction
- Magic methods

## 🛠️ Build
### LLM Client Architecture
Designed an extensible client hierarchy:

```text
LLMClient
├── GeminiClient
└── OpenAIClient
```

The design demonstrates how a common interface can support multiple LLM providers.

## 🧠 Key Takeaways
- Classes model real application components.
- Inheritance enables reuse.
- Polymorphism allows different implementations behind a common interface.
- Encapsulation keeps implementation details controlled.
- Abstraction helps build provider-independent systems.

## 🚀 Why It Matters for GenAI
Real GenAI applications frequently integrate multiple models, providers and tools. Good OOP design makes those systems easier to extend and maintain.
