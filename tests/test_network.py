import unittest

import numpy as np

from fashion_nn import NeuralNetwork, SoftmaxCrossEntropy


class NetworkTests(unittest.TestCase):
    def test_probabilities_have_expected_shape_and_sum_to_one(self) -> None:
        model = NeuralNetwork((4, 5, 3), seed=1)
        probabilities = model.predict_proba(np.ones((2, 4)))
        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_softmax_loss_is_finite(self) -> None:
        objective = SoftmaxCrossEntropy()
        probabilities, loss = objective.forward(
            np.array([[1000.0, 1001.0, 999.0]]), np.array([1])
        )
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(probabilities.sum(), 1.0)

    def test_training_learns_small_separable_dataset(self) -> None:
        inputs = np.array(
            [
                [-2.0, -1.0],
                [-1.5, -2.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.5, 2.0],
                [2.0, 1.0],
            ]
        )
        targets = np.array([0, 0, 0, 1, 1, 1])
        model = NeuralNetwork((2, 8, 2), seed=4)
        model.fit(inputs, targets, epochs=80, batch_size=6, learning_rate=0.05)
        self.assertEqual(float(np.mean(model.predict(inputs) == targets)), 1.0)


if __name__ == "__main__":
    unittest.main()
