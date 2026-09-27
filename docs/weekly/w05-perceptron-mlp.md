# W05 - Perceptron, Delta Rule, MLP, Backpropagation

**Topic:** Ch. 3 - Perceptron, Delta Rule, ANN, Backpropagation  
**Implementation Depth:** A (Build)  

---

## 1. Concept Capsule

The **Perceptron** models a single artificial neuron: it computes a weighted sum of inputs $z = \mathbf{w}^T\mathbf{x} + b$, passes it through a step activation, and outputs a binary class label. It learns via the **Delta Rule**, an error-correction mechanism that updates weights only on misclassification. The Perceptron Convergence Theorem guarantees convergence if and only if the data is linearly separable.

The **Multi-Layer Perceptron (MLP)** stacks multiple layers of neurons with non-linear activation functions (sigmoid, ReLU) between them. This enables learning non-linearly separable decision boundaries. Training uses **Backpropagation**: the chain rule applied layer-by-layer from output to input to compute gradients of the loss with respect to every weight.

**Role of non-linearity:** Without non-linear activations, any composition of linear layers collapses to a single linear transformation ($W_2(W_1 x) = W' x$). Non-linear activations break this collapse, giving the network the capacity to approximate arbitrary functions (Universal Approximation Theorem).

## 2. Derivation

### 2.1 Perceptron Delta Rule

The Perceptron predicts $\hat{y}^{(i)} \in \{0, 1\}$ using $z = \mathbf{w}^T\mathbf{x} + b$ and the step function:

$$
\phi(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

The weight update rule:

$$
\Delta w_j = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_j^{(i)}
$$

$$
w_j := w_j + \Delta w_j
$$

**Geometric interpretation:** The weight vector $\mathbf{w}$ is the normal to the decision hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$. On a false negative ($y=1, \hat{y}=0$), the update $\mathbf{w} \leftarrow \mathbf{w} + \eta\mathbf{x}$ rotates the hyperplane toward the misclassified point. On a false positive, it rotates away.

### 2.2 MLP Forward Pass

For a network with $L$ layers, the forward pass computes:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}, \quad a^{(l)} = \phi(z^{(l)})
$$

where $a^{(0)} = X$ (input) and $a^{(L)} = \text{softmax}(z^{(L)})$ (output).

### 2.3 Backpropagation (Chain Rule)

Given categorical cross-entropy loss $L = -\frac{1}{N}\sum y \log \hat{y}$, the gradients are computed backwards:

1. **Output error:** $\delta^{(L)} = a^{(L)} - y$ (softmax + cross-entropy simplification)
2. **Hidden layer error:** $\delta^{(l)} = (\delta^{(l+1)} {W^{(l+1)}}^T) \odot \phi'(z^{(l)})$
3. **Weight gradients:** $\frac{\partial L}{\partial W^{(l)}} = \frac{1}{N} {a^{(l-1)}}^T \delta^{(l)}$
4. **Bias gradients:** $\frac{\partial L}{\partial b^{(l)}} = \frac{1}{N} \sum \delta^{(l)}$

## 3. Code-to-Theory Trace

### Perceptron (`src/from_scratch/perceptron.py`)

The Delta Rule maps directly to:

```python
update = self.learning_rate * (target - self.predict(xi))
self.w_ += update * xi
self.b_ += update
```

### MLP (`src/from_scratch/mlp.py`)

The forward pass maps to `_forward()`:

```python
for l in range(n_layers):
    z = a @ self.weights_[l] + self.biases_[l]     # z^(l) = a^(l-1) W + b
    a = self._act_fn(z) if l < n_layers - 1 else softmax(z)
```

The backward pass maps to `_backward()`:

```python
delta = a_cache[-1] - y_onehot                      # Step 1: output error
for l in reversed(range(n_layers)):
    dW[l] = (a_cache[l].T @ delta) / n               # Step 3: weight gradient
    db[l] = np.sum(delta, axis=0, keepdims=True) / n  # Step 4: bias gradient
    if l > 0:
        delta = (delta @ self.weights_[l].T) * self._act_deriv(a_cache[l])  # Step 2
```

**Numerical gradient check** (`numerical_gradient_check()`) validates correctness using finite differences. Result: relative error $\approx 3 \times 10^{-10}$, confirming the backprop implementation is correct.

## 4. Experiment

### 4.1 Perceptron: Binary Classification (Walking vs Laying)

*Table 1: Perceptron binary classification on UCI HAR (Walking vs Laying, normalized, $\eta = 0.01$, 20 epochs). Source: `src/baseline/perceptron_eval.py`.*
<a id="table-1"></a>

| Model | Train Acc | Test Acc | Epochs to Converge |
|:---|:---|:---|:---|
| Custom Perceptron | $1.0000$ | $1.0000$ | 2 |
| Scikit-Learn Perceptron | $1.0000$ | $1.0000$ | N/A |

Walking vs Laying is highly linearly separable in the 561-dimensional feature space, confirming the Perceptron works correctly on separable data.

### 4.2 MLP: Capacity Experiment (Full 6-class UCI HAR)

*Table 2: MLP capacity experiment on UCI HAR (6-class, normalized, ReLU, $\eta = 0.01$, 100 epochs, full-batch GD). Source: `src/baseline/mlp_eval.py`.*
<a id="table-2"></a>

| Architecture | Train Acc | Test Acc | Test Macro-F1 |
|:---|:---|:---|:---|
| Small [32] | $0.8587$ | $0.8178$ | $0.8134$ |
| Medium [128] | $0.8566$ | $0.8324$ | $0.8272$ |
| Large [256, 128] | $0.9051$ | $0.8778$ | $0.8753$ |
| Sklearn MLP (128, SGD) | - | $0.9494$ | $0.9496$ |

**Observations:**

- Increasing capacity (more hidden units / deeper layers) improves performance, but the train-test gap also increases, signaling overfitting risk.
- Sklearn's MLP outperforms our implementation at the same architecture because it uses momentum, adaptive learning rate schedules, and mini-batch SGD, whereas our implementation uses vanilla full-batch gradient descent.

### 4.3 L2 Regularization Effect

*Table 3: Effect of L2 regularization on MLP [128] (same config as [Table 2](#table-2)). Source: `src/baseline/mlp_eval.py`.*
<a id="table-3"></a>

| $\lambda$ | Train Acc | Test Acc | Gap |
|:---|:---|:---|:---|
| $0.0$ | $0.8566$ | $0.8324$ | $0.0243$ |
| $10^{-4}$ | $0.8566$ | $0.8324$ | $0.0243$ |
| $10^{-3}$ | $0.8565$ | $0.8324$ | $0.0241$ |
| $10^{-2}$ | $0.8562$ | $0.8327$ | $0.0235$ |

L2 regularization slightly reduces the generalization gap. The effect is modest because 100 epochs of full-batch GD does not overfit severely in this setting.

## 5. Failure / Misconception

**The XOR Problem:** The Perceptron cannot learn XOR because the two classes are not linearly separable. My Perceptron code oscillates indefinitely on XOR unless stopped by `max_iter`. However, my MLP with a single hidden layer of 8 sigmoid units solves XOR perfectly (predictions: `[0, 1, 1, 0]`), demonstrating that a non-linear hidden layer is both necessary and sufficient to break linear inseparability.

## 6. Exam Capsule

The Perceptron is a linear binary classifier that updates weights via the Delta Rule: $\Delta w = \eta(y - \hat{y})x$. The Perceptron Convergence Theorem guarantees convergence on linearly separable data, but the algorithm diverges on non-separable data (e.g., XOR). Multi-Layer Perceptrons overcome this limitation by introducing hidden layers with non-linear activations. Training uses Backpropagation, which applies the chain rule layer-by-layer to compute $\partial L / \partial W^{(l)}$. Key practical considerations include weight initialization (He/Xavier), regularization (L2 weight decay, dropout), and learning rate scheduling. Increasing network capacity (width/depth) improves expressiveness but risks overfitting without regularization.

## 7. Reflection

- **What I can explain now:** The complete forward-backward pipeline, from input through hidden activations to softmax output, and back through the chain rule to weight gradients. I can also verify my gradients numerically.
- **Uncertainties:** Mini-batch SGD with momentum produces significantly better results (sklearn reaches 94.9% vs our 83.2%). Understanding adaptive optimizers (Adam, RMSProp) and their interaction with learning rate schedules is a clear next step.
- **Next steps:** Investigate why mini-batch + momentum is so much more effective than full-batch GD on this dataset; explore dropout as an alternative regularization strategy.

## 8. Inquiry Trail

- AI was used to scaffold the initial Perceptron and MLP class structure.
- I verified correctness by: (1) running the numerical gradient check (relative error ~3e-10), (2) confirming the Perceptron matches sklearn on UCI HAR, and (3) confirming the MLP solves XOR (which the Perceptron cannot).