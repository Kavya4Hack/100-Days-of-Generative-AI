import torch


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def predict(model, image):
    """Return the predicted class index and class name for one image tensor."""

    model.eval()

    # Add batch dimension if a single image is provided as [1, 28, 28].
    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad():
        logits = model(image)
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class, CLASS_NAMES[predicted_class]
