# Pre-Release Catch-Up

*As the assignment was officially released in Week 5, this document serves as a retrospective, comprehensive review of the theoretical foundations from Weeks 1-4. It formally records the extensive mathematical derivations, metric formulations, and algorithmic mechanics (specifically for Decision Trees) covered in the slides, alongside industry best practices. It also includes the required baseline diagnostic implementations.*

---

## 1. Machine Learning Foundations & Evaluation

### 1.1 Validation Strategies and Bootstrap Estimation

Robust evaluation is critical. The fundamental assumption is that data is Independent and Identically Distributed (i.i.d).

| Strategy | Mechanism | Use Case | Mathematical Note |
|:---|:---|:---|:---|
| **Hold-out** | Single static split (e.g., $70\%$ Train, $15\%$ Validation, $15\%$ Test). | Large datasets. | Highly sensitive to the initial random split. |
| **$k$-Fold CV** | Dataset partitioned into $k$ approximately equally sized folds. Each fold is used once for validation while the remaining $k-1$ folds are used for training. | Small to medium datasets. | Provides a more stable estimate of generalization performance than a single hold-out split. |
| **Bootstrap** | Resampling with replacement $B$ times. | Estimating uncertainty. | Probability of a sample being Out-Of-Bag (OOB) approaches $P(\text{OOB}) = \left(1-\frac{1}{n}\right)^n \rightarrow e^{-1} \approx 0.368$. |

Thus, approximately $36.8\%$ of observations are Out-Of-Bag in a given bootstrap sample.

### 1.2 The Bias-Variance Decomposition

The expected squared prediction error for a model $\hat{f}(x)$ predicting $Y$ given $X=x$ is expanded as:

$$ \mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}} $$

- **$\sigma^2$**: Irreducible noise inherent in the data.
- **Bias$^2$**: Error resulting from systematic approximation assumptions. High bias is typically associated with underfitting.
- **Variance**: Sensitivity to training-set fluctuations. High variance is typically associated with overfitting.

---

## 2. Baseline Pipeline and Leakage Checklist

Applying these theories requires strict adherence to a clean pipeline to prevent data leakage. The diagnostic pipeline for this curriculum enforces the following checklist:

- **Splitting First**: The dataset must be divided strictly prior to any feature scaling (e.g., Standardization) or imputation.
- **Isolating Transformations**: The transformation fitted on the training data is applied to validation and test data without refitting.
- **Group/Temporal Awareness**: 
    - *Group Data*: Ensuring repeated measurements from a single entity are wholly contained in either the train or test set.
    - *Temporal Data*: Ensuring future observations do not leak into training (e.g., $\text{Train}: t_1,\ldots,t_{80}$ and $\text{Test}: t_{81},\ldots,t_{100}$).
- **Cross-Validation Sealing**: Validating hyperparameter choices strictly inside the training fold using stratified splitting.

---

## 3. Model Taxonomy and Metrics

The machine learning algorithms investigated throughout this curriculum can be categorized into the following taxonomy:

| Model Family | Examples | Expected Bias/Variance Profile |
|:---|:---|:---|
| **Linear / Parametric** | Perceptron, Logistic Regression | Typically lower variance and potentially higher bias when the underlying decision boundary is substantially nonlinear. |
| **Non-Parametric** | Decision Trees, k-Nearest Neighbors | Highly dependent on hyperparameters. Small $k$ or deep trees yield low bias/high variance; large $k$ or shallow trees yield higher bias/lower variance. |
| **Probabilistic** | Naive Bayes, Hidden Markov Models | Depends on independence assumptions and parameter estimation. |
| **Ensemble Methods** | Random Forests, AdaBoost | Bagging/Random Forests primarily yield variance reduction; Boosting often yields bias reduction, though the full effect depends on algorithm settings. |

### 3.1 Performance Metrics

Choosing appropriate metrics is imperative, particularly when handling imbalanced class distributions.

- **Classification**: 
    - *Accuracy*: (Misleading for imbalanced data) $Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$
    - *Precision*: (Critical when false positives are costly) $Precision = \frac{TP}{TP+FP}$
    - *Recall*: (Critical when missing a positive is costly) $Recall = \frac{TP}{TP+FN}$
    - *F1-Score*: Harmonic mean of Precision and Recall $F_1 = 2 \frac{Precision \cdot Recall}{Precision + Recall}$

- **Regression**: 
    - *MSE / RMSE*: Emphasizes large deviations. 

        $$ MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2 $$
        
        $$ RMSE = \sqrt{MSE} $$

    - *$R^2$*: Measures the proportion of variance in the target explained by the model; $R^2=1$ indicates a perfect fit on the evaluated data (though this does not imply the model generalizes perfectly).

        $$R^2 = 1 - \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}$$


### 3.2 Train/Validation Curve Diagnosis

A critical theoretical concept for upcoming experiments is the interpretation of learning curves to diagnose the bias-variance trade-off. For future model training, I will apply the following diagnostic principles:

- **High Bias (Underfitting)**: Indicated when both the training error and validation error plateau at an unacceptably high value, suggesting the model lacks the capacity to capture the underlying data distribution.
- **High Variance (Overfitting)**: Indicated when the training error converges toward zero, but the validation error remains substantially higher than the training error and may begin to increase as model complexity or training progresses. This suggests the model is memorizing training noise rather than learning generalizable patterns.

---

## 4. Decision Trees: Theory

Decision trees recursively partition the feature space to maximize node purity.

### 4.1 Impurity Measures (ID3 vs C4.5 vs CART)

| Algorithm | Metric | Formulation | Key Characteristics |
|:---|:---|:---|:---|
| **ID3** | **Information Gain (IG)** | $IG(S, A) = H(S) - \sum_{v \in A} \frac{\|S_v\|}{\|S\|} H(S_v)$ <br><br> where $H(S) = - \sum p_i \log_2 p_i$ | Heavily biased toward attributes with many unique values. |
| **C4.5** | **Gain Ratio (GR)** | $GR(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$ <br><br> where $SplitInfo = - \sum \frac{\|S_v\|}{\|S\|} \log_2 \frac{\|S_v\|}{\|S\|}$ | Normalizes IG to penalize fragmentation (many small branches). Can create multiway splits. |
| **CART** | **Gini Index ($\Delta G$)** | $\Delta G = G(t) - \frac{N_L}{N_t}G(t_L) - \frac{N_R}{N_t}G(t_R)$ <br><br> where $G(t) = 1 - \sum p_i^2$ | Enforces strict **binary splits**. For regression, uses squared-error/variance reduction. |

### 4.2 Handling Real-World Complexities

- **Continuous Attributes**: Standard implementations sort unique values and test midpoints as candidate thresholds: $t_j = \frac{x_j+x_{j+1}}{2}$. They then evaluate splits as $x < t_j$ vs $x \ge t_j$.
- **Missing Values**:
    - *C4.5 Method*: Distributes the sample fractionally across child nodes using weighted fractional instances based on the observed distribution of the missing feature.
    - *CART Method*: The original CART methodology can use **surrogate splits**—alternative features whose splits closely reproduce the primary split—to route observations with missing values. Implementations (such as scikit-learn's) may instead use explicit imputation or other missing-value strategies.

### 4.3 Structural Regularization (Pruning)

Trees left unconstrained will overfit. Two primary regularization strategies exist:

1. **Pre-pruning (Early Stopping)**: Halting growth if specific thresholds are met, such as maximum depth, minimum samples per split, minimum samples per leaf, or minimum impurity decrease.
    - *Limitation*: Suffers from the "horizon effect," abandoning paths that yield low local gain but high subsequent gain.
2. **C4.5 Pessimistic Post-Pruning**: C4.5's pessimistic error pruning estimates an upper bound on the true error rate using a binomial confidence estimate, then compares the estimated error of a subtree with that of replacing it by a leaf.
3. **CART Cost-Complexity Pruning**: Defines an objective function $R_\alpha(T) = R(T) + \alpha |T|$, where $R(T)$ is the empirical risk / misclassification cost of tree $T$, $|T|$ is the number of terminal nodes, and $\alpha$ is the complexity parameter. The algorithm iteratively removes the weakest link, i.e., the internal node whose pruning produces the smallest increase in the objective per leaf removed. Cross-validation is used to select $\alpha$.

---

## 5. Baseline Diagnostic

To verify the theoretical mechanics of Decision Trees on a foundational level, I have implemented the required baseline components.

### 5.1 Impurity Routine Implementation

To evaluate node purity, I implemented the fundamental Shannon Entropy calculation from scratch:

```python title="src/from_scratch/impurity.py"
import numpy as np

def compute_entropy(y: np.ndarray) -> float:
    """Computes the Shannon entropy of a label distribution."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))
```

### 5.2 Tree Inspection

Training an unconstrained Decision Tree demonstrates the non-parametric nature of the model: it recursively partitions until all leaf nodes achieve absolute purity (zero entropy) or constraints are met. A structural trace of the initial splits from our Pre-pruned UCI HAR model is visualized below:

```text
|--- feature_52 <= 0.10
|   |--- class: 6
|--- feature_52 >  0.10
|   |--- feature_389 <= -0.97
|   |   |--- feature_559 <= 0.14
|   |   |   |--- truncated branch of depth 3
|   |   |--- feature_559 >  0.14
|   |   |   |--- truncated branch of depth 3
|   |--- feature_389 >  -0.97
|   |   |--- feature_508 <= -0.52
|   |   |   |--- truncated branch of depth 3
|   |   |--- feature_508 >  -0.52
|   |   |   |--- truncated branch of depth 3
```

### 5.3 Stopping vs. Pruning Comparison

I evaluated structural constraints on the UCI HAR dataset (Train: $7352$, Test: $2947$) with a fixed random seed (`42`). The experiment revealed a clear distinction between pre-pruning and post-pruning regularization:

| Architecture | Train Accuracy | Test Accuracy | Tree Depth | Leaves |
|:---|:---|:---|:---|:---|
| **Unconstrained** | $1.0000$ | $0.8626$ | $18$ | $163$ |
| **Pre-pruned** (`max_depth=5`) | $0.9204$ | $0.8385$ | $5$ | $17$ |
| **Post-pruned** (`ccp_alpha=0.002`) | $0.9543$ | $\mathbf{0.8694}$ | $9$ | $32$ |

- **Unconstrained Overfitting**: Without limits, the tree memorized the training set ($100\%$ accuracy) but exhibited high variance, generating a massive structure ($163$ leaves) that generalized less effectively.
- **Pre-pruning (Early Stopping)**: Aggressively limiting depth (to $5$) mitigated overfitting but introduced the "horizon effect." The model stopped learning prematurely, missing deeper, informative splits and resulting in lower overall performance ($83.85\%$).
- **Cost-Complexity Post-pruning**: By fully growing the tree and subsequently collapsing weakest-link subtrees ($\alpha=0.002$), the algorithm achieved the best test accuracy ($86.94\%$) while drastically reducing complexity (from $163$ to just $32$ leaves). This empirically confirms that evaluating split paths retrospectively is superior to greedy early stopping.

---

## 6. Academic Reflection

The curriculum from Weeks 1-4 establishes a mathematical foundation for understanding model fitting and generalization. The transition from probability-based concepts such as entropy to greedy decision-making in decision trees illustrates how statistical criteria can be translated into practical learning algorithms. While individual decision trees are highly interpretable, unconstrained trees can exhibit high variance, motivating regularization and pruning. These concepts provide an important foundation for understanding ensemble methods such as Random Forests and Gradient Boosting.
