# Day 12 — Forward Propagation, Loss & Backpropagation 🔁

## 🎯 Objective
Understand how a neural network produces predictions and learns from its errors.

## 📚 Concepts Covered
- Forward propagation
- Loss functions
- Backpropagation
- Neural-network mathematics

## 🔄 Training Flow

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
```

## 🧠 Forward Propagation
Inputs pass through the network to produce a prediction.

## 📉 Loss
The loss function measures how far the prediction is from the target.

## 🔙 Backpropagation
Backpropagation calculates how the parameters contributed to the error by propagating gradients backward through the network.

## 🧠 Key Takeaways
The central learning loop is:

**prediction → loss → gradients → parameter updates**

Understanding this mathematically makes PyTorch training loops much easier to reason about.
