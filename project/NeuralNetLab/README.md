# 🧠 NeuralNetLab --- Fashion-MNIST Neural Network Classifier

A PyTorch-based neural network classification project built as part of
my **100 Days of GenAI** journey.

The project demonstrates the complete workflow of building, training,
validating, evaluating, saving, and loading a neural network for image
classification.

------------------------------------------------------------------------

## 🚀 Project Overview

**NeuralNetLab** classifies grayscale clothing images from the
**Fashion-MNIST** dataset into one of 10 categories.

The model takes a `28 × 28` grayscale image as input and predicts one of
the following classes:

    Label Class
  ------- -------------
        0 T-shirt/top
        1 Trouser
        2 Pullover
        3 Dress
        4 Coat
        5 Sandal
        6 Shirt
        7 Sneaker
        8 Bag
        9 Ankle boot

The project is designed to build a strong foundation in **PyTorch,
neural networks, training loops, model evaluation, and ML project
structure**.

------------------------------------------------------------------------

## 🎯 Objectives

-   Understand PyTorch tensors and tensor shapes
-   Load and preprocess Fashion-MNIST
-   Build a neural network using `nn.Module` and `nn.Linear`
-   Understand forward propagation
-   Train a model using backpropagation
-   Use `CrossEntropyLoss`
-   Use the Adam optimizer
-   Split data into training and validation sets
-   Evaluate performance on unseen test data
-   Save and reload trained model weights
-   Build a reusable inference pipeline
-   Organize an ML project for GitHub and portfolio use

------------------------------------------------------------------------

## 🏗️ Model Architecture

The current neural network uses a fully connected architecture:

``` text
Input Image
28 × 28
   │
   ▼
Flatten
   │
784 features
   │
   ▼
Linear: 784 → 128
   │
   ▼
ReLU
   │
   ▼
Linear: 128 → 64
   │
   ▼
ReLU
   │
   ▼
Linear: 64 → 10
   │
   ▼
Class Logits
```

### Why 784 inputs?

Each image contains:

``` text
28 × 28 = 784 pixels
```

The image is flattened before being passed to the first fully connected
layer.

### Why 10 outputs?

Fashion-MNIST contains 10 classes, so the final layer produces 10
logits.

------------------------------------------------------------------------

## 📊 Dataset

This project uses **Fashion-MNIST**.

-   Training images: 60,000
-   Test images: 10,000
-   Image size: 28 × 28
-   Image type: grayscale
-   Number of classes: 10

The dataset is downloaded automatically using `torchvision`.

------------------------------------------------------------------------

## 🔄 ML Pipeline

``` text
Fashion-MNIST
      ↓
Data Loading
      ↓
ToTensor()
      ↓
Train / Validation Split
      ↓
DataLoader
      ↓
Neural Network
      ↓
Forward Pass
      ↓
Cross-Entropy Loss
      ↓
Backpropagation
      ↓
Adam Optimizer
      ↓
Validation
      ↓
Test Evaluation
      ↓
Model Saving
      ↓
Inference
```

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   PyTorch
-   Torchvision
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Google Colab
-   Git & GitHub

------------------------------------------------------------------------

## 📁 Project Structure

``` text
01-NeuralNetLab/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── NeuralNetLab.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── models/
    └── neuralnet_fashion_mnist.pth
```

------------------------------------------------------------------------

## 🧪 Training

The model uses:

``` python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

and:

``` python
loss_function = nn.CrossEntropyLoss()
```

The training process follows:

``` python
optimizer.zero_grad()

predictions = model(images)

loss = loss_function(predictions, labels)

loss.backward()

optimizer.step()
```

### Training logic

``` text
Clear old gradients
        ↓
Forward pass
        ↓
Calculate loss
        ↓
Backpropagation
        ↓
Update parameters
        ↓
Repeat for batches
        ↓
Repeat for epochs
```

------------------------------------------------------------------------

## 📈 Evaluation

The project evaluates the model using:

-   Training loss
-   Validation loss
-   Training accuracy
-   Validation accuracy
-   Test accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix

The training and validation curves are also used to identify possible
overfitting.

------------------------------------------------------------------------

## 💾 Model Saving

The trained model is saved using PyTorch's `state_dict`:

``` python
torch.save(
    model.state_dict(),
    "models/neuralnet_fashion_mnist.pth"
)
```

The model can later be reconstructed and loaded:

``` python
model = NeuralNetwork()

model.load_state_dict(
    torch.load("models/neuralnet_fashion_mnist.pth")
)

model.eval()
```

This allows inference without retraining the model.

------------------------------------------------------------------------

## 🔮 Inference

After loading the trained model, a new Fashion-MNIST image can be passed
through the model:

``` text
Input Image
    ↓
Preprocessing
    ↓
Neural Network
    ↓
10 Class Logits
    ↓
Highest Score
    ↓
Predicted Class
```

Example:

``` text
Prediction: Sneaker
```

------------------------------------------------------------------------

## 📌 Key Concepts Learned

### PyTorch

-   Tensor
-   Tensor shape
-   Dataset
-   DataLoader
-   `nn.Module`
-   `nn.Linear`
-   ReLU
-   Model parameters
-   `state_dict`

### Deep Learning

-   Forward propagation
-   Loss function
-   Gradients
-   Backpropagation
-   Optimization
-   Learning rate
-   Epoch
-   Batch
-   Training vs validation
-   Overfitting

### Machine Learning

-   Train/validation/test split
-   Classification
-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix
-   Generalization

------------------------------------------------------------------------

## 🔧 Future Improvements

This project will be progressively improved as part of the 100-Day GenAI
journey.

Planned improvements include:

-   [ ] Hyperparameter tuning
-   [ ] Compare different hidden-layer sizes
-   [ ] Learning-rate experiments
-   [ ] Dropout
-   [ ] Batch normalization
-   [ ] Early stopping
-   [ ] Better visualization
-   [ ] Interactive prediction interface
-   [ ] REST API for inference
-   [ ] Dockerization
-   [ ] Automated testing
-   [ ] CI/CD

------------------------------------------------------------------------

## 🎓 Part of 100 Days of GenAI

This project is **Project 1** in my 100-Day GenAI learning journey.

The broader progression is:

``` text
Neural Network
      ↓
NLP
      ↓
Transformer
      ↓
LLM
      ↓
RAG
      ↓
AI Agents
      ↓
Multi-Agent Systems
      ↓
Full-Stack GenAI System
```

The purpose of this project is to establish the deep-learning and
PyTorch foundations required for the later projects.

------------------------------------------------------------------------

## 👨‍💻 Author

**Kavya**

This project was built as part of a hands-on journey toward becoming a
**Generative AI Engineer**.

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.
