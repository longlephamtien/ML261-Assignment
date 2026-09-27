# CO3117 - Machine Learning: Individual Longitudinal Assignment

## Use Case

I use the [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset to predict physical activity from smartphone inertial sensors. The dataset, split, and metric are frozen from R0 onward.

- **Primary metric:** Macro-F1
- **Split:** subject-aware (no person in both train and test)
- **Test set:** sealed for final comparison
- **Seeds:** 42, (123, 456)

## Timeline

- **Part I** (40%) - due 14 Oct 2026: Foundations, Decision Tree, Perceptron/MLP, Naive Bayes, GA, BN/TAN
- **Part II** (60%) - due 2 days before final: HMM, SVM, PCA/LDA, Ensemble, Logistic/MaxEnt/CRF, synthesis

## Setup

Requires Python >= 3.12. Key dependencies: NumPy, scikit-learn, pandas, matplotlib, MkDocs Material.

```bash
git clone https://github.com/longlephamtien/ML261-Assignment
cd ML261-Assignment
uv sync
uv run python src/data.py --download
uv run pytest tests/
uv run mkdocs serve
```

## Study Stack

- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [numpy-ml](https://github.com/ddbourgin/numpy-ml)
- [pyprobml](https://github.com/probml/pyprobml)
- [NPTEL IIT Madras](https://nptel.ac.in/courses/106106139)
- [scikit-learn](https://scikit-learn.org)

See [REFERENCES.md](REFERENCES.md) for full citations. AI usage is logged in [AI_USE.md](AI_USE.md).
