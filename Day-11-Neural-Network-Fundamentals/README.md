# Day 11 — Neural Network Fundamentals 🧠

## 🎯 Objective
Understand the building blocks of neural networks before implementing them with PyTorch.

## 📚 Concepts Covered
- Neurons
- Weights
- Bias
- Activation functions
- Neural-network intuition

## 🧠 Basic Neuron

```text
Inputs
  ↓
Weighted Sum + Bias
  ↓
Activation Function
  ↓
Output
```

Mathematically:

```text
z = w₁x₁ + w₂x₂ + ... + b

output = activation(z)
```

## 🧠 Key Takeaways
- Weights determine how strongly inputs influence a neuron.
- Bias shifts the activation.
- Activation functions introduce non-linearity.
- Neural networks learn parameters from data.

## 🚀 Why It Matters for GenAI
Modern language models are built from neural-network components. Understanding neurons and learned parameters is the foundation for understanding Transformers and LLMs.
