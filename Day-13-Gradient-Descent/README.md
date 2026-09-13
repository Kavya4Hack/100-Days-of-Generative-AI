# Day 13 — Gradient Descent 📉

## 🎯 Objective
Understand how neural networks update their parameters to reduce prediction error.

## 🔄 Core Process

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Weight Update
    ↓
Better Prediction
    ↓
Repeat
```

## 📐 Core Idea

A parameter is updated in the direction that reduces the loss:

```text
new parameter = old parameter - learning rate × gradient
```

## 📚 Concepts Covered
- Gradient
- Learning rate
- Weight updates
- Loss minimization
- Iterative optimization

## 🧠 Key Takeaways
- The gradient tells us the direction of increasing loss.
- We move in the opposite direction to reduce loss.
- The learning rate controls the size of the update.
- Repeated updates gradually optimize the model.

## 🚀 Why It Matters for GenAI
Training large neural networks and LLMs is fundamentally an optimization problem. Gradient-based learning is one of the core ideas behind modern deep learning.
