"""A small fully connected neural network implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class Dense:
    """A trainable fully connected layer."""

    def __init__(self, inputs: int, neurons: int, rng: np.random.Generator) -> None:
        self.weights = rng.standard_normal((inputs, neurons)) * np.sqrt(2.0 / inputs)
        self.biases = np.zeros((1, neurons))
        self.weight_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.biases)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases
        return self.output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.weight_gradient = self.inputs.T @ gradient
        self.bias_gradient = np.sum(gradient, axis=0, keepdims=True)
        return gradient @ self.weights.T

    def update(self, learning_rate: float, momentum: float) -> None:
        self.weight_velocity = (
            momentum * self.weight_velocity - learning_rate * self.weight_gradient
        )
        self.bias_velocity = (
            momentum * self.bias_velocity - learning_rate * self.bias_gradient
        )
        self.weights += self.weight_velocity
        self.biases += self.bias_velocity


class ReLU:
    """Rectified linear activation."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0.0, inputs)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient * (self.inputs > 0)


class SoftmaxCrossEntropy:
    """Numerically stable softmax activation and categorical loss."""

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        self.probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        self.targets = targets
        clipped = np.clip(self.probabilities, 1e-7, 1 - 1e-7)
        losses = -np.log(clipped[np.arange(len(clipped)), targets])
        return self.probabilities, float(np.mean(losses))

    def backward(self) -> np.ndarray:
        gradient = self.probabilities.copy()
        gradient[np.arange(len(gradient)), self.targets] -= 1
        return gradient / len(gradient)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    learning_rate: float
    loss: float
    accuracy: float


class NeuralNetwork:
    """Configurable multilayer classifier using dense layers and ReLU."""

    def __init__(self, dimensions: Iterable[int], seed: int = 42) -> None:
        dimensions = tuple(dimensions)
        if len(dimensions) < 2:
            raise ValueError("dimensions must include input and output sizes")
        rng = np.random.default_rng(seed)
        self.layers = [
            Dense(dimensions[index], dimensions[index + 1], rng)
            for index in range(len(dimensions) - 1)
        ]
        self.activations = [ReLU() for _ in self.layers[:-1]]
        self.objective = SoftmaxCrossEntropy()

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs
        for layer, activation in zip(self.layers[:-1], self.activations):
            output = activation.forward(layer.forward(output))
        return self.layers[-1].forward(output)

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.logits(inputs)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(inputs), axis=1)

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        momentum: float,
    ) -> tuple[float, float]:
        probabilities, loss = self.objective.forward(self.logits(inputs), targets)
        accuracy = float(np.mean(np.argmax(probabilities, axis=1) == targets))

        gradient = self.objective.backward()
        gradient = self.layers[-1].backward(gradient)
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            gradient = layer.backward(activation.backward(gradient))

        for layer in self.layers:
            layer.update(learning_rate, momentum)
        return loss, accuracy

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.02,
        momentum: float = 0.9,
        seed: int = 42,
    ) -> list[EpochMetrics]:
        rng = np.random.default_rng(seed)
        history: list[EpochMetrics] = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(inputs))
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            batches = 0
            current_rate = max(0.001, learning_rate * (0.95 ** ((epoch - 1) // 5)))
            for start in range(0, len(inputs), batch_size):
                selection = order[start : start + batch_size]
                loss, accuracy = self.train_batch(
                    inputs[selection], targets[selection], current_rate, momentum
                )
                epoch_loss += loss
                epoch_accuracy += accuracy
                batches += 1
            history.append(
                EpochMetrics(
                    epoch=epoch,
                    learning_rate=current_rate,
                    loss=epoch_loss / batches,
                    accuracy=epoch_accuracy / batches,
                )
            )
        return history
