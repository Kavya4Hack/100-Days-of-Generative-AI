# Day 16 — PyTorch Training Loop 🔄

## 🎯 Objective
Implement and understand the complete PyTorch training loop.

## 📚 Concepts Covered
- Model initialization
- Loss function
- Optimizer
- Forward pass
- Loss calculation
- `zero_grad()`
- Backpropagation
- `loss.backward()`
- Optimizer step
- Epoch-based training

## 🔄 Training Loop

```python
for epoch in range(epochs):

    optimizer.zero_grad()

    predictions = model(X)

    loss = loss_function(predictions, y)

    loss.backward()

    optimizer.step()
```

## 🧠 Why Each Step Exists

| Step | Purpose |
|---|---|
| `zero_grad()` | Clear old gradients |
| Forward pass | Generate predictions |
| Loss | Measure error |
| `backward()` | Compute gradients |
| `step()` | Update parameters |

## 🧠 Key Takeaways
The training loop is the heart of supervised neural-network training.

Understanding it is more important than memorizing framework syntax.
