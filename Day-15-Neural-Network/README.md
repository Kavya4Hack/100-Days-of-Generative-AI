# Day 15 — Build a Neural Network 🧠⚡

## 🎯 Objective
Build a neural network in PyTorch and understand how its components fit together.

## 📚 Concepts Covered
- `torch.nn`
- Layers
- Parameters
- Forward pass
- Loss function
- Optimizer
- Backpropagation
- Parameter updates

## 🏗️ Model Concept

```text
Input
  ↓
Linear Layer
  ↓
Activation
  ↓
Linear Layer
  ↓
Output
```

## 🔄 Training Concept

```text
Input
 ↓
Model
 ↓
Prediction
 ↓
Loss
 ↓
loss.backward()
 ↓
Optimizer
 ↓
Updated Parameters
```

## 🧠 Key Takeaways
This day connected the mathematical ideas from Days 11–13 with actual PyTorch code.

The important distinction is:
- `forward()` computes predictions.
- The loss measures error.
- `backward()` computes gradients.
- The optimizer updates parameters.
