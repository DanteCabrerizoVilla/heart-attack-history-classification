# Heart Attack History Classification

Academic Data Science / Machine Learning project focused on classifying patients according to whether they report a previous heart attack, using general health, demographic and lifestyle variables.

The project was developed as an introductory end-to-end ML exercise, covering data exploration, class imbalance, preprocessing, model comparison, interpretability and exploratory clustering.

> **Important:** `HadHeartAttack` represents a previous heart attack reported in the dataset. The model therefore classifies historical records; it is **not** a prospective clinical risk predictor or a medical decision-making tool.

## Project overview

The dataset contains **237,630 records and 35 variables**, with **13,201 positive cases (~5.6%)** for `HadHeartAttack`. The strong class imbalance makes accuracy alone misleading, so the analysis focuses on metrics such as ROC AUC, F1, recall, balanced accuracy and average precision.

The project follows this workflow:

1. Exploratory data analysis
2. Class imbalance analysis
3. Train/test split and preprocessing with `Pipeline`
4. Comparison of baseline, Logistic Regression, Decision Tree, Random Forest and XGBoost
5. Cross-validation and hyperparameter search
6. Final evaluation on the test set
7. Model interpretation with SHAP and LIME
8. Exploratory clustering with K-Means
9. Discussion of limitations and possible next steps

## Main results

In the current experiment, several models perform similarly. Logistic Regression and Random Forest are closely matched in ROC AUC, with XGBoost also very close.

On the held-out test set:

| Model | ROC AUC | F1 | Recall | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.883 | 0.793 | 0.768 | 0.800 |
| XGBoost | 0.881 | 0.794 | 0.790 | 0.795 |
| Random Forest (tuned) | 0.881 | 0.795 | 0.791 | 0.796 |
| Random Forest | 0.881 | 0.796 | 0.789 | 0.797 |
| Decision Tree | 0.842 | 0.772 | 0.760 | 0.776 |

These metrics were obtained after balancing the dataset through random undersampling. Because the test set in this experiment is also based on the balanced sample, the results should be interpreted as **model comparisons within this experiment**, not as estimates of performance at the original 5.6% prevalence.

One of the main lessons of the project was that the initial **~94% accuracy** was misleading in the presence of severe class imbalance; the first version achieved a recall of only **0.22** for the positive class.

## Interpretability

SHAP and LIME are used to examine how the trained model arrives at its predictions.

The strongest contributions are concentrated around variables such as `HadAngina` and `ChestScan`. Rather than treating these variables as causal risk factors, the notebook uses this result to question whether some predictors may contain information related to medical attention occurring after the heart attack.

This is one of the main reasons the project explicitly discusses the limitations of the dataset and target definition.

## Clustering

K-Means is included as an exploratory analysis to identify groups of patients with similar encoded characteristics.

The clustering section is deliberately presented as exploratory rather than as a validated risk segmentation system. In particular, the chosen `k` is discussed in relation to silhouette-based evaluation and interpretability.

## Limitations

This is an academic project and has several important methodological limitations:

- The target represents a **previous heart attack**, not a future event.
- The dataset is observational and based on self-reported health information.
- Some highly influential variables may reflect information obtained after the event.
- Random undersampling discards a large number of negative examples.
- The balanced test set does not reproduce the original population prevalence.
- Feature-selection decisions in the original academic workflow were simplified rather than embedded in a fully nested validation procedure.
- The encoding of some ordinal/categorical variables is simplified.
- Probabilities were not calibrated and the classification threshold was kept at 0.5.

These limitations are intentionally kept visible because they are part of the learning process of the project.

## What I would do next

A stronger version of the analysis could:

- keep the test set at the original prevalence and apply resampling only to the training data;
- compare undersampling with approaches such as `class_weight`;
- optimise the decision threshold according to the cost of false positives and false negatives;
- repeat the modelling after removing variables that may contain post-event information;
- investigate calibration and probability quality.

## Repository structure

```text
heart-attack-history-classification/
├── notebooks/
│   └── heart_attack_history_classification.ipynb
├── datos/
│   └── Patients Data.xlsx
├── README.md
└── requirements.txt
```

## How to run

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook notebooks/heart_attack_history_classification.ipynb
```

The notebook expects the dataset at:

```text
datos/Patients Data.xlsx
```

If the dataset is stored elsewhere, update `DATA_PATH` in the first code section.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn
- SHAP
- LIME
