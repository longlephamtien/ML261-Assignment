"""
Multi-Layer Perceptron (MLP) implemented from scratch using NumPy.

Covers the full forward pass, backpropagation (chain rule),
and numerical gradient checking as required by the CO3117 spec.

Reference: CO3117 ANN slides (MLP.pdf, TrainingANN.pdf)
"""

import numpy as np


# ============================================================================
# Activation functions
# ============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation: sigma(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given the activation output a = sigma(z).
    sigma'(z) = sigma(z) * (1 - sigma(z)) = a * (1 - a).
    """
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, z)."""
    return np.maximum(0, z)


def relu_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of ReLU given the activation output."""
    return (a > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for output layer."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}


# ============================================================================
# Loss functions
# ============================================================================

def cross_entropy_loss(y_true_onehot: np.ndarray, y_pred: np.ndarray) -> float:
    """Categorical cross-entropy loss.
    L = -1/N * sum( y * log(y_hat) )
    """
    eps = 1e-12
    n = y_true_onehot.shape[0]
    return -np.sum(y_true_onehot * np.log(y_pred + eps)) / n


# ============================================================================
# MLP class
# ============================================================================

class MLP:
    """
    Multi-Layer Perceptron with explicit forward and backward passes.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons in each layer, including input and output.
        Example: [561, 128, 64, 6] means 561 inputs, two hidden layers
        (128 and 64 neurons), and 6 output classes.
    activation : str
        Activation function for hidden layers ('sigmoid' or 'relu').
    learning_rate : float
        Step size for gradient descent.
    max_iter : int
        Number of training epochs.
    random_state : int
        Seed for reproducibility.
    l2_lambda : float
        L2 regularization strength (weight decay). 0.0 = no regularization.

    Attributes
    ----------
    weights_ : list[np.ndarray]
        Weight matrices W^(l) for each layer transition.
    biases_ : list[np.ndarray]
        Bias vectors b^(l) for each layer.
    loss_history_ : list[float]
        Training loss recorded at each epoch.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        max_iter: int = 100,
        random_state: int = 42,
        l2_lambda: float = 0.0,
    ):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.l2_lambda = l2_lambda

        self.weights_ = []
        self.biases_ = []
        self.loss_history_ = []

        self._act_fn, self._act_deriv = ACTIVATIONS[activation]

    def _init_weights(self):
        """The initialization for ReLU, Xavier for sigmoid."""
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = []
        self.biases_ = []

        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            if self.activation == "relu":
                # He initialization: var = 2 / fan_in
                std = np.sqrt(2.0 / fan_in)
            else:
                # Xavier initialization: var = 2 / (fan_in + fan_out)
                std = np.sqrt(2.0 / (fan_in + fan_out))

            W = rgen.normal(0.0, std, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))

            self.weights_.append(W)
            self.biases_.append(b)

    def _onehot(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """Convert integer labels to one-hot encoding."""
        onehot = np.zeros((y.shape[0], n_classes))
        onehot[np.arange(y.shape[0]), y] = 1.0
        return onehot

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Explicit forward pass through all layers.

        For each layer l:
            z^(l) = a^(l-1) @ W^(l) + b^(l)
            a^(l) = phi(z^(l))          [hidden layers]
            a^(L) = softmax(z^(L))      [output layer]

        Returns
        -------
        z_cache : list of pre-activation values (z) for each layer
        a_cache : list of post-activation values (a) for each layer,
                  starting with a[0] = X (input)
        """
        z_cache = []
        a_cache = [X]  # a^(0) = input

        a = X
        n_layers = len(self.weights_)

        for l in range(n_layers):
            z = a @ self.weights_[l] + self.biases_[l]
            z_cache.append(z)

            if l < n_layers - 1:
                # Hidden layer: apply activation
                a = self._act_fn(z)
            else:
                # Output layer: softmax
                a = softmax(z)

            a_cache.append(a)

        return z_cache, a_cache

    # ------------------------------------------------------------------
    # Backward pass (Backpropagation)
    # ------------------------------------------------------------------

    def _backward(
        self,
        y_onehot: np.ndarray,
        z_cache: list[np.ndarray],
        a_cache: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Explicit backward pass using the chain rule.

        Step 1: Compute output error
            delta^(L) = a^(L) - y   (softmax + cross-entropy simplification)

        Step 2: Backpropagate through hidden layers
            delta^(l) = (delta^(l+1) @ W^(l+1).T) * phi'(z^(l))

        Step 3: Compute gradients
            dW^(l) = (1/N) * a^(l-1).T @ delta^(l) + lambda * W^(l)
            db^(l) = (1/N) * sum(delta^(l))

        Returns
        -------
        dW : list of weight gradients
        db : list of bias gradients
        """
        n = y_onehot.shape[0]
        n_layers = len(self.weights_)

        dW = [None] * n_layers
        db = [None] * n_layers

        # --- Output layer error ---
        # For softmax + cross-entropy, the gradient simplifies elegantly:
        delta = a_cache[-1] - y_onehot  # shape: (N, n_classes)

        for l in reversed(range(n_layers)):
            # Gradient for weights and biases at layer l
            dW[l] = (a_cache[l].T @ delta) / n
            db[l] = np.sum(delta, axis=0, keepdims=True) / n

            # L2 regularization gradient
            if self.l2_lambda > 0:
                dW[l] += self.l2_lambda * self.weights_[l]

            # Propagate error to previous layer (skip for input layer)
            if l > 0:
                delta = (delta @ self.weights_[l].T) * self._act_deriv(a_cache[l])

        return dW, db

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        """
        Train the MLP using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,), integer class labels

        Returns
        -------
        self
        """
        self._init_weights()

        n_classes = self.layer_sizes[-1]
        y_onehot = self._onehot(y, n_classes)

        for epoch in range(self.max_iter):
            # Forward
            z_cache, a_cache = self._forward(X)

            # Loss
            loss = cross_entropy_loss(y_onehot, a_cache[-1])
            if self.l2_lambda > 0:
                l2_term = sum(np.sum(W ** 2) for W in self.weights_)
                loss += 0.5 * self.l2_lambda * l2_term
            self.loss_history_.append(loss)

            # Backward
            dW, db = self._backward(y_onehot, z_cache, a_cache)

            # Update weights (gradient descent)
            for l in range(len(self.weights_)):
                self.weights_[l] -= self.learning_rate * dW[l]
                self.biases_[l] -= self.learning_rate * db[l]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        _, a_cache = self._forward(X)
        return a_cache[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ============================================================================
# Numerical gradient checking
# ============================================================================

def numerical_gradient_check(
    mlp: MLP,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5,
) -> float:
    """
    Verify backprop gradients against numerical (finite-difference) gradients.

    For each parameter theta_i:
        numerical_grad = (L(theta_i + eps) - L(theta_i - eps)) / (2 * eps)

    Then compute relative error:
        rel_err = ||grad_analytic - grad_numerical|| / (||grad_analytic|| + ||grad_numerical|| + eps)

    A relative error < 1e-5 indicates correct backprop.

    Returns
    -------
    max_relative_error : float
    """
    n_classes = mlp.layer_sizes[-1]
    y_onehot = mlp._onehot(y, n_classes)

    # Get analytic gradients from backprop
    z_cache, a_cache = mlp._forward(X)
    dW_analytic, db_analytic = mlp._backward(y_onehot, z_cache, a_cache)

    max_rel_err = 0.0

    # Check weight gradients for each layer
    for l in range(len(mlp.weights_)):
        W = mlp.weights_[l]
        grad_numerical = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                old_val = W[i, j]

                # f(theta + eps)
                W[i, j] = old_val + epsilon
                _, a_plus = mlp._forward(X)
                loss_plus = cross_entropy_loss(y_onehot, a_plus[-1])

                # f(theta - eps)
                W[i, j] = old_val - epsilon
                _, a_minus = mlp._forward(X)
                loss_minus = cross_entropy_loss(y_onehot, a_minus[-1])

                # Restore
                W[i, j] = old_val

                grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        # Relative error
        diff = np.linalg.norm(dW_analytic[l] - grad_numerical)
        norm_sum = np.linalg.norm(dW_analytic[l]) + np.linalg.norm(grad_numerical) + 1e-8
        rel_err = diff / norm_sum
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  Layer {l} weight gradient relative error: {rel_err:.2e}")

    return max_rel_err


# ============================================================================
# Sanity tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: XOR problem (non-linearly separable)")
    print("=" * 60)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    mlp_xor = MLP(
        layer_sizes=[2, 8, 2],
        activation="sigmoid",
        learning_rate=1.0,
        max_iter=2000,
        random_state=42,
    )
    mlp_xor.fit(X_xor, y_xor)

    preds = mlp_xor.predict(X_xor)
    print(f"XOR Predictions: {preds} (expected: [0, 1, 1, 0])")
    print(f"XOR Correct: {np.all(preds == y_xor)}")
    print(f"Final loss: {mlp_xor.loss_history_[-1]:.6f}")

    print()
    print("=" * 60)
    print("Test 2: Numerical gradient check (small network)")
    print("=" * 60)

    mlp_check = MLP(
        layer_sizes=[2, 4, 2],
        activation="sigmoid",
        learning_rate=0.1,
        max_iter=1,
        random_state=42,
    )
    mlp_check._init_weights()

    print("Running numerical gradient check...")
    max_err = numerical_gradient_check(mlp_check, X_xor, y_xor)
    print(f"Max relative error: {max_err:.2e}")
    if max_err < 1e-5:
        print("PASS: Backpropagation gradients are correct.")
    else:
        print("FAIL: Gradient mismatch detected.")
