# Day 19 — Neural Network Build 🏗️

## 🎯 Objective
Apply the PyTorch concepts learned so far to build a working neural-network model and inspect its training behavior.

## 📚 Concepts Applied
- Tensors
- Model definition
- Linear layers
- Forward propagation
- Loss functions
- Backpropagation
- Optimizer
- Training loop
- Predictions
- Loss monitoring

## 🔄 End-to-End Flow

```text
Training Data
     ↓
Tensor
     ↓
Neural Network
     ↓
Prediction
     ↓
Loss
     ↓
Backpropagation
     ↓
Weight Update
     ↓
Repeat
```

## 🧪 Training Observation
The model's loss was monitored across training iterations to observe whether learning was occurring.

## 🧠 Key Takeaways
- A neural network learns by adjusting parameters.
- Loss should generally decrease when the training process is working correctly.
- Training output is useful for diagnosing model behavior.
- Understanding the mechanics is more important than treating the model as a black box.

## 🚀 Next Step
Use the same building blocks to create a complete classification system with training, validation, evaluation and model saving.
