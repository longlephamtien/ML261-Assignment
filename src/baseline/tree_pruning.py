import sys
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data import get_data

def main():
    print("Loading UCI HAR dataset...")
    # Load data without normalization since trees don't require scaled features
    data = get_data(normalize=False)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print("-" * 50)
    
    # 1. Unconstrained Tree (High Variance / Overfitting)
    print("Training Unconstrained Decision Tree...")
    tree_unconstrained = DecisionTreeClassifier(random_state=42)
    tree_unconstrained.fit(X_train, y_train)
    
    train_acc_unc = accuracy_score(y_train, tree_unconstrained.predict(X_train))
    test_acc_unc = accuracy_score(y_test, tree_unconstrained.predict(X_test))
    
    print(f"Unconstrained - Train Accuracy: {train_acc_unc:.4f}")
    print(f"Unconstrained - Test Accuracy:  {test_acc_unc:.4f}")
    print(f"Tree Depth: {tree_unconstrained.get_depth()}")
    print(f"Number of Leaves: {tree_unconstrained.get_n_leaves()}")
    print("-" * 50)
    
    # 2. Pre-pruned Tree (Early Stopping)
    print("Training Pre-pruned Tree (max_depth=5)...")
    tree_pre = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_pre.fit(X_train, y_train)
    
    train_acc_pre = accuracy_score(y_train, tree_pre.predict(X_train))
    test_acc_pre = accuracy_score(y_test, tree_pre.predict(X_test))
    
    print(f"Pre-pruned    - Train Accuracy: {train_acc_pre:.4f}")
    print(f"Pre-pruned    - Test Accuracy:  {test_acc_pre:.4f}")
    print(f"Tree Depth: {tree_pre.get_depth()}")
    print(f"Number of Leaves: {tree_pre.get_n_leaves()}")
    print("-" * 50)
    
    # 3. Post-pruned Tree (Cost-Complexity Pruning)
    print("Training Post-pruned Tree (ccp_alpha=0.002)...")
    tree_post = DecisionTreeClassifier(random_state=42, ccp_alpha=0.002)
    tree_post.fit(X_train, y_train)
    
    train_acc_post = accuracy_score(y_train, tree_post.predict(X_train))
    test_acc_post = accuracy_score(y_test, tree_post.predict(X_test))
    
    print(f"Post-pruned   - Train Accuracy: {train_acc_post:.4f}")
    print(f"Post-pruned   - Test Accuracy:  {test_acc_post:.4f}")
    print(f"Tree Depth: {tree_post.get_depth()}")
    print(f"Number of Leaves: {tree_post.get_n_leaves()}")
    print("-" * 50)

if __name__ == "__main__":
    main()
