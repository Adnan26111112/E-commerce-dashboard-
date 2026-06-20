# 🛒 E-Commerce Sales Performance & Business Insights Dashboard

> End-to-end data analysis project on 100,000+ real orders from **Olist**, a Brazilian e-commerce marketplace — covering data cleaning, exploratory analysis, machine learning, time series forecasting, and a deployed interactive dashboard.

**🔗 Live Dashboard:** [Add your Streamlit Cloud link here once deployed]

---

## 📌 The Business Problem

Olist's core challenge isn't acquiring customers — it's keeping them. Only a small fraction of customers ever place a second order. This project investigates **why**, predicts **who** is likely to churn, and forecasts **what** revenue will look like over the next 6 months — using nothing but raw transactional data.

---

## 🔑 Key Findings

*(Fill these in with your actual numbers after running the notebooks)*

| Finding | Business Implication |
|---|---|
| Repeat purchase rate is **X.X%** | Retention, not acquisition, is the biggest growth lever |
| Review scores drop sharply after **X** days of delivery | A delivery SLA of ≤X days is a measurable retention driver |
| Churn model achieves **X.XX ROC-AUC** (5-fold CV) | Review score and delivery time are the top predictors of repeat purchase |
| 6-month revenue forecast: **R$X.XM** (MAPE: X.X%) | Forecast informs inventory, staffing, and budget planning |
| São Paulo accounts for **~X%** of revenue | Geographic concentration risk — underserved states are a growth opportunity |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Wrangling | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (Random Forest, Logistic Regression) |
| Forecasting | Facebook Prophet |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📊 What's Inside

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Cleaned and merged 8 relational tables (orders, items, payments, reviews, customers, products, sellers) into one master dataset
- Revenue trends, category performance, customer behavior, delivery analysis, RFM segmentation, geographic and time-pattern analysis

### 2. Machine Learning — Churn Prediction (`02_ml_churn.ipynb`)
- Binary classification: will a customer return after their first order?
- Logistic Regression baseline → Random Forest main model
- Leak-proof feature engineering (features built only from first-order data)
- Evaluated with ROC-AUC, precision/recall, 5-fold cross-validation
- Feature importance via permutation testing — translated into business recommendations

### 3. Revenue Forecasting (`03_forecasting.ipynb`)
- Facebook Prophet time series model on monthly revenue
- Chronological train/test split (no data leakage across time)
- Trend + seasonality decomposition
- 6-month forward forecast with 95% confidence intervals

### 4. Interactive Dashboard (`app.py`)
Six-tab Streamlit app:
- **Executive Summary** — top-line KPIs, revenue trend, RFM segments
- **Sales Performance** — filterable revenue trends, time patterns, geographic breakdown
- **Customer Analytics** — purchase frequency, delivery-vs-review analysis, RFM detail
- **Product Insights** — category performance, payment analysis, bubble chart
- **ML Predictions** — model metrics, feature importance, **live churn predictor**
- **Revenue Forecast** — 6-month projection with confidence bands

---

## 📁 Project Structure

```
ecommerce-dashboard/
├── .streamlit/
│   └── config.toml              # dashboard theme config
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ml_churn.ipynb
│   └── 03_forecasting.ipynb
├── models/
│   ├── churn_model.pkl
│   ├── feature_cols.pkl
│   └── scaler.pkl
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── master.csv                   # cleaned, merged dataset
├── rfm.csv
├── forecast.csv
├── monthly_revenue.csv
├── model_predictions.csv
├── feature_importance.csv
└── README.md
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/YOUR-USERNAME/ecommerce-dashboard.git
cd ecommerce-dashboard
pip install -r requirements.txt
streamlit run app.py
```

The notebooks must be run in order (01 → 02 → 03) first if you want to regenerate the data files from the raw [Olist dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

---

## 🧠 Decisions I Made (and why)

- **Filtered to `delivered` orders only** for revenue analysis — cancelled/unavailable orders don't represent real revenue and would distort KPIs.
- **Built churn features only from first-order data** — using any information from later orders would leak future information into a model meant to predict at the moment of first purchase.
- **Chose Random Forest over Logistic Regression** after confirming it meaningfully beat the linear baseline on cross-validated ROC-AUC — the relationship between delivery experience and retention isn't purely linear.
- **Used permutation importance over built-in feature importance** — built-in importance is biased toward high-cardinality features; permutation importance gives a more honest signal of what the model actually relies on.
- **Split the forecasting time series chronologically, not randomly** — random splitting on time series data leaks future information into training, which would make evaluation meaningless.

---

## 📈 Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100,000+ real orders made at multiple marketplaces in Brazil between 2016–2018, across 8 relational CSV files (orders, items, payments, reviews, customers, products, sellers).

---

## 👤 Author

**Adnan**
Data Analysis Portfolio Project
[LinkedIn](#) · [GitHub](#)
