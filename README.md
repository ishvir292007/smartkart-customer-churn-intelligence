# 🛒 SmartKart Customer Churn Intelligence

### An Interpretable Machine Learning System for Predictive Customer Retention

SmartKart Customer Churn Intelligence is an end-to-end **machine learning pipeline for customer churn prediction**. The project transforms messy customer data into actionable churn-risk insights using data cleaning, outlier treatment, feature selection, feature standardisation, and **Logistic Regression**.

The objective is not only to predict **which customers are likely to churn**, but also to provide a business-oriented risk report that can help a retention team prioritise customers for intervention.

---

## 🎯 Business Problem

Customer churn directly impacts revenue, customer lifetime value, and retention costs.

For SmartKart, the goal is to identify customers who are at higher risk of leaving **before churn occurs**, allowing the business to take proactive retention actions.

### Business Question

> **Which SmartKart customers are most likely to churn, and what customer characteristics are associated with higher churn risk?**

The model uses three business-relevant predictors:

* **Age**
* **Monthly Spend**
* **Complaints**

The target variable is:

* `0` → No Churn
* `1` → Churn

---

## 🚀 Project Highlights

* End-to-end supervised machine learning workflow
* Real-world-style messy customer dataset
* Data quality inspection and cleaning
* Duplicate detection and removal
* Missing-value treatment using median imputation
* Invalid-value detection and correction
* IQR-based outlier treatment
* Feature selection based on business relevance
* Train-test split with stratification
* Feature standardisation
* Logistic Regression classification
* Churn probability scoring
* Confusion matrix analysis
* Precision, Recall, F1-score and Accuracy evaluation
* Interpretable model coefficients
* Business-ready customer risk report
* Ranked identification of high-risk customers

---

## 🧠 Machine Learning Pipeline

The project follows a structured **15-step ML lifecycle**:

```text
Data Collection
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Outlier Detection & Treatment
      ↓
Feature Selection
      ↓
Target Definition
      ↓
Target Encoding Verification
      ↓
Train-Test Split
      ↓
Feature Standardisation
      ↓
Model Building
      ↓
Model Training
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Model Interpretation
      ↓
Business-Ready Output
```

---

## 📊 Dataset

The original dataset contains **100 customer records and 5 columns**:

| Feature         | Description                   | Role       |
| --------------- | ----------------------------- | ---------- |
| `Customer_ID`   | Unique customer identifier    | Identifier |
| `Age`           | Customer age                  | Feature    |
| `Monthly_Spend` | Customer's monthly spending   | Feature    |
| `Complaints`    | Number of customer complaints | Feature    |
| `Churn`         | Whether the customer churned  | Target     |

The dataset was intentionally designed with real-world-style data-quality problems, including:

* Missing values
* Duplicate records
* Leading/trailing whitespace
* Text values inside numeric fields
* Invalid ages
* Negative spending values
* Extreme spending outliers
* Extreme complaint counts

This makes the project representative of a practical **data preprocessing + machine learning workflow** rather than a clean toy dataset.

---

## 🧹 Data Preprocessing

### 1. Duplicate Removal

Duplicate customer records were identified and removed before modelling.

The dataset was reduced from:

```text
100 records → 95 records
```

### 2. Data Type Correction

The `Age` column contained inconsistent values such as:

```text
" 25 "
"thirty"
-5
150
```

These were converted into a consistent numeric representation, with invalid ages treated as missing.

### 3. Missing Value Treatment

Missing values in:

* `Age`
* `Monthly_Spend`
* `Complaints`

were handled using **median imputation**.

The median was selected because it is more robust to extreme values than the mean.

### 4. Invalid Value Handling

Negative monthly spending was treated as invalid because customer spending cannot logically be negative.

### 5. Outlier Treatment

The **Interquartile Range (IQR)** method was used to identify extreme observations.

Instead of deleting potentially useful customer records, extreme values were capped using winsorisation.

---

## 🔍 Feature Selection

`Customer_ID` was intentionally excluded from the model because it is an identifier rather than a meaningful behavioural predictor.

The final feature set was:

```python
[
    "Age",
    "Monthly_Spend",
    "Complaints"
]
```

These variables represent customer characteristics and behavioural indicators that can potentially explain churn.

---

## 🎯 Target Variable

The target variable is:

```text
Churn
```

with:

```text
0 = No Churn
1 = Churn
```

The dataset contains a relatively balanced target distribution, making it suitable for binary classification.

---

## ⚙️ Model Architecture

### Algorithm

**Logistic Regression**

Logistic Regression was selected because churn is a **binary classification problem** and the model provides both:

1. A predicted churn class
2. A probability of churn

The probability output is particularly useful for business applications because customers can be ranked according to their estimated churn risk.

---

## 🧪 Model Training

The cleaned dataset was divided into:

```text
80% Training Data
20% Testing Data
```

A stratified split was used to preserve the churn distribution across training and testing datasets.

Features were standardised using `StandardScaler`.

Importantly, the scaler was fitted only on the training data and then applied to the test data to avoid test-set information leakage.

---

## 📈 Model Performance

The trained Logistic Regression model achieved the following results on the held-out test set:

| Metric        |      Result |
| ------------- | ----------: |
| **Accuracy**  |  **89.47%** |
| **Precision** |  **83.33%** |
| **Recall**    | **100.00%** |
| **F1-Score**  |  **90.91%** |

The confusion matrix was:

```text
[[7, 2],
 [0, 10]]
```

This means the model correctly identified all actual churners in the test set, resulting in **100% recall**.

For a retention use case, high recall is valuable because failing to identify a customer who is genuinely at risk can result in a preventable customer loss.

> **Important:** These results are based on a small 19-customer test set, so they should be interpreted as a project-level demonstration rather than evidence of production-grade generalisation.

---

## 🔬 Model Interpretability

One of the advantages of Logistic Regression is its interpretability.

The trained model produced the following coefficients:

```text
Age             +0.650
Monthly_Spend   -2.170
Complaints      +1.621
```

### Interpretation

#### 💰 Monthly Spend

`Monthly_Spend` has the strongest negative coefficient.

Higher monthly spending is associated with **lower predicted churn risk** within this dataset.

#### 📞 Complaints

`Complaints` has a strong positive coefficient.

More complaints are associated with **higher predicted churn risk**.

This provides a clear potential business intervention point: improving customer service and complaint resolution may support retention.

#### 👤 Age

`Age` has a smaller positive coefficient compared with the other two variables.

The model therefore assigns relatively less influence to age than to monthly spending and complaints.

---

## 💡 Key Business Insights

The model suggests two major retention signals:

### 1. Customer Complaints

Customers with higher complaint levels tend to show higher churn risk.

**Potential business action:**

* Prioritise customers with repeated complaints
* Improve complaint-resolution time
* Introduce proactive service recovery
* Escalate unresolved customer issues

### 2. High-Value Customers

Higher monthly spend is associated with lower churn risk in this dataset.

**Potential business action:**

* Protect relationships with high-value customers
* Monitor changes in their behaviour
* Develop loyalty and retention programmes
* Identify sudden changes in spending patterns

---

## 📋 Business-Ready Output

The project converts model predictions into an actionable customer risk report.

Each prediction contains:

```text
Customer_ID
Age
Monthly_Spend
Complaints
Actual_Churn
Predicted_Churn
Churn_Probability
Risk_Label
```

Customers are ranked by:

```text
Churn_Probability
```

from highest to lowest.

The resulting report can be used by a hypothetical retention team to prioritise customers requiring intervention.

Example risk classification:

```text
High predicted probability
        ↓
Likely to Churn
        ↓
Retention intervention
```

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualisation

* Matplotlib
* Seaborn

### Environment

* Google Colab / Jupyter Notebook

---

## 📁 Project Structure

```text
smartkart-customer-churn-intelligence/
│
├── SmartKart_Churn_Prediction_ML_Pipeline.ipynb
│
├── SmartKart_dirty_100_rows.csv
│
├── smartkart_churn_risk_report.csv
│
└── README.md
```

---

## ▶️ How to Run

### Option 1 — Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Upload the SmartKart CSV dataset when prompted.
3. Run the notebook from top to bottom.
4. Review the preprocessing and model outputs.
5. Download the generated churn-risk report.

### Option 2 — Jupyter Notebook

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then launch:

```bash
jupyter notebook
```

Open:

```text
SmartKart_Churn_Prediction_ML_Pipeline.ipynb
```

and execute the cells sequentially.

---

## 📌 Limitations

This project is designed as an ML learning and portfolio implementation using a **small 100-record dataset**.

Therefore:

* The dataset is too small for production deployment.
* Model performance may not generalise to a larger customer population.
* Only three predictive features are currently used.
* No temporal customer behaviour is incorporated.
* No hyperparameter optimisation or model benchmarking is included.
* Production deployment, monitoring, and retraining are outside the current scope.

A production implementation would require substantially more historical customer data and additional behavioural, transactional, demographic, and engagement features.

---

## 🔮 Future Improvements

Potential next steps include:

* Expand the dataset with real customer histories
* Add purchase frequency and recency
* Add customer tenure
* Add average order value
* Add discount/coupon usage
* Add website/app engagement
* Add customer-support interaction history
* Add payment behaviour
* Compare Logistic Regression with Random Forest, XGBoost and other classifiers
* Perform cross-validation
* Perform hyperparameter optimisation
* Calibrate churn probabilities
* Develop a real-time churn scoring API
* Build a customer retention dashboard
* Deploy the model using FastAPI/Flask
* Containerise the application with Docker
* Implement model monitoring and retraining

---

## 📊 Project Outcome

This project demonstrates the complete transformation of raw customer data into a business-oriented machine learning solution:

```text
Messy Customer Data
        ↓
Data Quality Management
        ↓
Feature Engineering & Selection
        ↓
Machine Learning
        ↓
Churn Probability
        ↓
Customer Risk Ranking
        ↓
Retention Decision Support
```

The final system demonstrates how machine learning can move beyond prediction and translate customer data into **actionable retention intelligence**.

---

## 👨‍💻 Skills Demonstrated

This project demonstrates practical experience with:

* Python
* Pandas
* NumPy
* Data Cleaning
* Exploratory Data Analysis
* Missing Value Handling
* Outlier Detection
* Feature Selection
* Feature Standardisation
* Supervised Learning
* Binary Classification
* Logistic Regression
* Model Evaluation
* Confusion Matrix
* Precision & Recall
* F1-Score
* Model Interpretability
* Churn Analytics
* Business Intelligence
* Predictive Customer Retention

---

## 📜 Project Context

**Course:** Introduction to AI & ML
**Programme:** BBA AI/ML
**Institution:** Chitkara Business School
**Learning Objective:** Apply data preprocessing, feature selection and machine learning models to business scenarios and evaluate model performance using appropriate metrics.

---

## ⭐ Author

**Ishvir Singh Matharoo**

BBA FinTech & AI
Chitkara University

---

### ⭐ If you found this project useful

Feel free to explore the notebook, experiment with different classification algorithms, and extend the system into a production-ready customer retention platform.
