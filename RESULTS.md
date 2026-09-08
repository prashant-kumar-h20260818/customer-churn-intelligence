# Validated Model Results

These results were produced by the repository's GitHub Actions pipeline on the IBM Telco Customer Churn dataset using Python 3.11 and the code currently on `main`.

## Dataset and split

- Total customers: **7,043**
- Historical churn rate: **26.54%**
- Training customers: **5,634**
- Held-out test customers: **1,409**
- Split: **80/20 stratified**
- Model selection: **5-fold stratified cross-validation** on the training set only

## Cross-validation model comparison

| Model | Mean CV ROC-AUC | CV Std. Dev. |
|---|---:|---:|
| **Logistic Regression** | **0.8469** | 0.0109 |
| Random Forest | 0.8459 | 0.0089 |
| XGBoost | 0.8433 | 0.0092 |

Logistic Regression was selected because it achieved the highest mean cross-validation ROC-AUC.

## Held-out test performance

The operating threshold was selected from out-of-fold training predictions by maximizing **F2**, which intentionally weights churn recall more strongly than precision.

| Metric | Value |
|---|---:|
| Selected model | **Logistic Regression** |
| Decision threshold | **0.34** |
| ROC-AUC | **0.8446** |
| PR-AUC | **0.6493** |
| Recall | **0.9091** |
| Precision | 0.4450 |
| F1 | 0.5975 |
| F2 | **0.7522** |
| Accuracy | 0.6749 |

### Confusion matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 611 | 424 |
| Actual Churn | 34 | 340 |

The recall-optimized policy identifies **340 of 374** actual churners in the held-out set and misses **34**, corresponding to **90.91% churn recall**. The trade-off is a larger false-positive retention pool, which is why production threshold selection should incorporate intervention cost, campaign capacity, margin, and expected save value.

## Leading global model drivers

Because Logistic Regression was selected, global importance is represented by absolute standardized coefficient magnitude. The strongest transformed features included:

1. Two-year contract
2. Fiber-optic internet service
3. Month-to-month contract
4. DSL internet service
5. Monthly charges
6. Tenure
7. Total number of services
8. Paperless billing / billing configuration signals
9. Multiple-lines configuration
10. Technical-support / internet-service signals

These are predictive associations in this dataset, not causal effects.

## Reproducibility

The same validation can be rerun with:

```bash
python -m pytest -q
python -m src.train
```

GitHub Actions also executes both steps on pushes to `main` and uploads the generated model metrics and feature-importance outputs as a workflow artifact.

## Resume-safe quantified statement

> Built an end-to-end churn intelligence system across **7,043 customers and 20+ source attributes**, benchmarking Logistic Regression, Random Forest and XGBoost with 5-fold cross-validation; selected Logistic Regression achieved **0.845 held-out ROC-AUC and 90.9% churn recall** at a recall-optimized threshold.

Do not claim that the model reduced churn or increased retention revenue: this dataset contains no treatment/control retention experiment from which causal business uplift could be measured.
