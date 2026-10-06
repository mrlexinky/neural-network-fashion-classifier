"""Command-line training entry point."""

from __future__ import annotations

import argparse

from .data import load_fashion_csv
from .network import NeuralNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NumPy fashion classifier")
    parser.add_argument("dataset", help="label-first Fashion-MNIST CSV file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    images, labels = load_fashion_csv(args.dataset, args.limit)
    model = NeuralNetwork((784, 64, 32, 10))
    for metrics in model.fit(images, labels, epochs=args.epochs):
        print(
            f"Epoch {metrics.epoch:02d} | LR {metrics.learning_rate:.4f} | "
            f"loss {metrics.loss:.4f} | accuracy {metrics.accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
