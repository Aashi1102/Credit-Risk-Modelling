# 💳 Credit Risk Modeling

A machine learning project that predicts the probability of loan default using customer demographics, loan details, and credit bureau data. Built following industry-standard credit scorecard methodology (WOE/IV, VIF, rank-ordering validation) rather than a plain "fit and predict" approach.

---

## 📌 Problem Statement

Lenders need to know, **before disbursing a loan, how likely a customer is to default.** This project builds an end-to-end pipeline — from raw customer/loan/bureau data to a trained, validated, production-ready risk model — that outputs a probability of default for each applicant.

---

## 📂 Project Structure

```
credit-risk-model/
├── credit_risk_model.ipynb     # Main notebook: EDA, feature engineering, modeling, evaluation
├── dataset/                     # Raw input data
│   ├── customers.csv
│   ├── loans.csv
│   └── bureau_data.csv
├── artifacts/
│   └── model_data.joblib        # Saved final model + scaler + feature list
├── App/                         # Deployment app (prediction interface)
└── SOW Credit Risk Model.pdf    # Scope of work / project brief
```

---

## 🗃️ Dataset

Three source files, joined on `cust_id`:

| File | Contents |
|---|---|
| `customers.csv` | Demographics — age, gender, income, marital status, residence type, etc. |
| `loans.csv` | Loan details — loan amount, tenure, purpose, sanction amount, processing fee, etc. |
| `bureau_data.csv` | Credit bureau history — open/closed accounts, delinquent months, DPD, credit utilization, enquiry count |

**Target variable:** `default` (1 = customer defaulted, 0 = customer repaid)

The dataset has significant **class imbalance** (far fewer defaults than non-defaults), which is explicitly handled during modeling.

---

## 🔧 Methodology

The notebook follows a disciplined, leakage-free pipeline:

1. **Train/Test split performed before EDA** — to prevent any decision (imputation values, outlier thresholds, feature choices) from being influenced by test data.
2. **Data cleaning** — missing value imputation (mode for categoricals), duplicate removal, typo correction in categorical labels.
3. **Outlier removal via business rules** — e.g., processing fee capped at 3% of loan amount, GST capped at 20%, net disbursement ≤ loan amount — validated against the same rules on the test set.
4. **Exploratory Data Analysis** — boxplots, histograms, and KDE plots segmented by default status to identify visually strong predictors.
5. **Feature Engineering** — derived ratio-based features that carry more signal than raw values:
   - `loan_to_income` (LTI)
   - `delinquency_ratio`
   - `avg_dpd_per_delinquency`
6. **Multicollinearity check (VIF)** — dropped redundant numeric features (`sanction_amount`, `processing_fee`, `gst`, `net_disbursement`, `principal_outstanding`).
7. **Feature selection (WOE & IV)** — Weight of Evidence / Information Value calculated for every feature; only features with **IV > 0.02** retained — a standard credit-scoring industry threshold.
8. **Encoding** — one-hot encoding (`drop_first=True`) for categorical variables.
9. **Class imbalance handling** — compared Random Under-sampling vs. **SMOTETomek** (synthetic oversampling + boundary cleaning).
10. **Hyperparameter tuning** — RandomizedSearchCV baseline, then **Optuna** for more efficient, guided search.
11. **Model evaluation** — Classification report, ROC-AUC, Gini coefficient, KS statistic, and decile-wise rank-ordering analysis (not just accuracy — critical for imbalanced classification problems).

---

## 🤖 Models Tried

| Attempt | Model | Imbalance Handling | Tuning |
|---|---|---|---|
| 1 | Logistic Regression, Random Forest, XGBoost | None (baseline) | — |
| 2 | Logistic Regression, XGBoost | Random Under-sampling | — |
| 3 | Logistic Regression | SMOTETomek | Optuna |
| 4 | XGBoost | SMOTETomek | Optuna |

**Final model:** Tuned **Logistic Regression** — chosen over XGBoost despite comparable performance, because logistic regression coefficients are directly interpretable, which is essential for regulatory and business explainability in credit decisioning.

---

## 📊 Model Performance (Final Model)

| Metric | Value |
|---|---|
| AUC-ROC | **0.98** |
| Gini Coefficient | **0.96** |
| Rank Ordering | Confirmed across deciles (KS statistic) |

An AUC of 0.98 and Gini of 0.96 indicate the model separates defaulters from non-defaulters with near-perfect discriminative power, and predicted risk deciles show consistent rank ordering — validated the same way real-world credit bureaus (e.g., CIBIL) evaluate scorecards.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn, XGBoost
- **Imbalance handling:** imbalanced-learn (SMOTETomek, RandomUnderSampler)
- **Hyperparameter tuning:** Optuna
- **Statistics:** statsmodels (VIF)
- **Model persistence:** joblib

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn optuna statsmodels joblib
```

### Run
1. Place `customers.csv`, `loans.csv`, and `bureau_data.csv` inside the `dataset/` folder.
2. Open and run `credit_risk_model.ipynb` cell by cell.
3. The final trained model bundle is saved to `artifacts/model_data.joblib` and contains:
   - the trained model
   - the fitted scaler
   - the list of feature columns
   - the list of columns requiring scaling

This bundle can be loaded directly for inference on new applicant data without retraining.

---

## 📈 Future Improvements

- Deploy the saved model as a REST API / web app for real-time scoring (see `App/`)
- Add SHAP-based explainability for individual predictions
- Periodic model monitoring and retraining as new loan performance data comes in
- Experiment with monotonic constraints on tree-based models for regulatory-friendly non-linear modeling

---

## 📄 License

This project is for educational and portfolio purposes.
