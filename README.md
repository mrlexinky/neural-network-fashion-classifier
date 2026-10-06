# Neural Network for Fashion Classification

![Project presentation showing a classified T-shirt](assets/project-overview.png)

A fully connected neural network built from first principles with Python and NumPy. I developed it for my Extended Project Qualification (EPQ) to understand the mathematics and engineering behind image classification rather than relying on TensorFlow, PyTorch or a prebuilt training framework.

The project classifies 28 x 28 grayscale Fashion-MNIST images across ten clothing categories. It implements dense layers, ReLU and softmax activations, cross-entropy loss, backpropagation, mini-batch training, momentum and learning-rate scheduling.

## Why this project matters

This was my first long-form independent software project. It combined mathematical research with iterative development, testing, performance tuning and written evaluation. The final implementation grew from a small network that classified synthetic coordinates into a configurable image-classification pipeline trained on 60,000 examples.

## Results

| Measure | Recorded result | Interpretation |
| --- | ---: | --- |
| Time to 95% training accuracy | 2 min 5 sec | Met the original under-five-minute target |
| Highest reported training accuracy | 99.97% | Achieved after 17 min 33 sec in a longer run |
| Saved 50-epoch notebook run | 94.25% | Final epoch recorded in the included notebook |

These figures are **training accuracy**, not held-out test accuracy. The original project did not build a rigorous validation pipeline, so the 99.97% result may indicate overfitting. External-image preprocessing was also inconsistent. I describe both limitations in [the project case study](docs/PROJECT.md#evaluation-and-limitations).

![Testing evidence from the final presentation](assets/testing.png)

## Technical overview

The default architecture is `784 -> 64 -> 32 -> 10`:

1. Flatten and normalise each 28 x 28 image.
2. Apply two dense hidden layers with ReLU activation.
3. Produce class probabilities with softmax.
4. Calculate categorical cross-entropy loss.
5. Backpropagate analytical gradients.
6. Update weights and biases using momentum-based gradient descent.

The implementation uses NumPy for numerical arrays and matrix multiplication, but the network, training loop and gradient calculations are written directly in this repository.

## Repository guide

```text
src/fashion_nn/           Reusable neural-network implementation
tests/                    Fast tests for shapes, gradients and learning
notebooks/                Sanitised EPQ development notebook
docs/PROJECT.md           Full project case study and evaluation
CITATIONS.md              Complete project bibliography
assets/                   Original presentation and test images
```

## Run it

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

To train on Fashion-MNIST, download `fashion-mnist_train.csv` from the Fashion-MNIST project or Kaggle, then run:

```bash
python -m fashion_nn.train path/to/fashion-mnist_train.csv --epochs 10
```

Use `--limit 5000` for a quick local demonstration. The dataset and trained pickle are deliberately excluded because they are large generated/external artefacts.

## What I learned

- How matrix dimensions flow through forward and backward passes.
- Why numerical stability matters in softmax and logarithmic loss.
- How batch size, learning rate, momentum and weight initialisation affect training.
- How to convert research into a working artefact, test it and evaluate its weaknesses honestly.

The development story, design choices and next steps are documented in [docs/PROJECT.md](docs/PROJECT.md). Every research source used in the EPQ is listed in [CITATIONS.md](CITATIONS.md).

## Licence

The code is available under the [MIT Licence](LICENSE). Presentation images and written project material remain copyright © Oscar Horton.
