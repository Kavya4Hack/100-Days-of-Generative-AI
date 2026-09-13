"""
Day 14 — PyTorch Fundamentals
Tensor, shape, device and operations.
"""

import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        device=device,
    )

    y = torch.tensor(
        [[1.0], [5.0], [9.0], [13.0], [17.0]],
        device=device,
    )

    print("Tensor:")
    print(x)

    print("\nShape:", x.shape)
    print("Device:", x.device)
    print("CUDA available:", torch.cuda.is_available())

    print("\nOperations:")
    print("x + 2 =", x + 2)
    print("x * 3 =", x * 3)
    print("x + y =", x + y)


if __name__ == "__main__":
    main()
