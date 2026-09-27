import numpy as np

class Perceptron:
    """
    Perceptron classifier implemented from scratch using NumPy.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate (eta) between 0.0 and 1.0.
    max_iter : int, default=50
        Passes over the training dataset (epochs).
    random_state : int, default=42
        Random number generator seed for random weight initialization.
        
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : float
        Bias unit after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 50, random_state: int = 42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.errors_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values. Assumes binary classification (e.g., 0 and 1).
            
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights randomly (small numbers) rather than all zeros
        # to prevent symmetric stagnation in some scenarios.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []

        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Delta rule: weight_change = learning_rate * (target - predicted)
                # If target == predicted, update is 0.
                update = self.learning_rate * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                
                # Count misclassifications
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step function"""
        # Step function: if net_input >= 0 return 1, else 0
        return np.where(self.net_input(X) >= 0.0, 1, 0)

if __name__ == "__main__":
    # Sanity check with a simple linearly separable logic gate (OR gate)
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_test = np.array([0, 1, 1, 1])
    
    ppn = Perceptron(learning_rate=0.1, max_iter=10)
    ppn.fit(X_test, y_test)
    
    print("Testing Perceptron on OR gate:")
    print("Predictions:", ppn.predict(X_test))
    print("Errors per epoch:", ppn.errors_)
    print("Final weights:", ppn.w_)
    print("Final bias:", ppn.b_)
