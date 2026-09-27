import numpy as np

def compute_entropy(y: np.ndarray) -> float:
    """
    Computes the Shannon entropy of a label distribution.
    
    Args:
        y: A 1D numpy array of class labels.
        
    Returns:
        The entropy value (float).
    """
    if len(y) == 0:
        return 0.0
        
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    
    # Add a small epsilon to prevent log2(0)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def compute_gini(y: np.ndarray) -> float:
    """
    Computes the Gini impurity of a label distribution.
    
    Args:
        y: A 1D numpy array of class labels.
        
    Returns:
        The Gini impurity value (float).
    """
    if len(y) == 0:
        return 0.0
        
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    
    return 1.0 - np.sum(probabilities ** 2)

if __name__ == "__main__":
    y_pure = np.array([1, 1, 1, 1])
    y_mixed = np.array([1, 1, 0, 0])
    
    print(f"Entropy of pure node: {compute_entropy(y_pure):.4f}")
    print(f"Entropy of mixed (50/50) node: {compute_entropy(y_mixed):.4f}")
    print(f"Gini of pure node: {compute_gini(y_pure):.4f}")
    print(f"Gini of mixed (50/50) node: {compute_gini(y_mixed):.4f}")
