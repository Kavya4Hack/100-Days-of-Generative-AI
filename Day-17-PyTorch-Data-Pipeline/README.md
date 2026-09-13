# Day 17 — PyTorch Data Pipeline 📦

## 🎯 Objective
Learn how PyTorch manages training data efficiently.

## 📚 Concepts Covered
- `Dataset`
- `DataLoader`
- Batching
- Epochs
- Learning rate

## 🗂️ Data Flow

```text
Dataset
   ↓
DataLoader
   ↓
Batches
   ↓
Training Loop
   ↓
Model
```

## 📦 Dataset
Represents the data and provides access to individual samples.

## 🚚 DataLoader
Loads samples in batches and makes iteration through the dataset easier.

## 🔁 Epoch
One complete pass through the training dataset.

## 📈 Learning Rate
Controls how large the optimizer's parameter updates are.

## 🧠 Key Takeaways
Data handling is part of model engineering. A model is only as useful as the pipeline that feeds it.

## 🚀 Why It Matters for GenAI
The same concepts scale to large datasets, fine-tuning pipelines and production ML workloads.
