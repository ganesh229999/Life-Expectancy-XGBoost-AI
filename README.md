# 🌍 Life Expectancy Intelligence

### *End-to-End Machine Learning Pipeline for Global Health Prediction*

<p align="center">

<img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/XGBoost-97.07%25-success?style=for-the-badge" />
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn" />
<img src="https://img.shields.io/badge/Desktop_App-Tkinter-informational?style=for-the-badge" />
<img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=for-the-badge" />

</p>

---

# 📌 Overview

**Life Expectancy Intelligence** is a production-ready Machine Learning project designed to predict **global life expectancy** using socio-economic, healthcare, and demographic indicators.

The project goes beyond basic ML experimentation and delivers:

✅ Complete Data Preprocessing Pipeline
✅ Statistical Feature Engineering
✅ Hyperparameter Tuned XGBoost Model
✅ Cross-Validation Based Evaluation
✅ Interactive Desktop GUI Application
✅ Real-Time AI Predictions

The final optimized model achieved an impressive:

# 🏆 **97.07% R² Accuracy**

using a tuned **XGBoost Regressor**.

---

# 🚀 Recruiter Highlights

<table>
<tr>
<td width="50%">

### ✅ Core Skills Demonstrated

* Machine Learning
* Data Science
* Predictive Analytics
* Feature Engineering
* Hyperparameter Tuning
* Cross Validation
* Statistical Analysis
* GUI Development
* Model Deployment

</td>

<td width="50%">

### ✅ Technologies Used

* Python
* XGBoost
* Scikit-Learn
* Pandas
* NumPy
* Tkinter
* Matplotlib
* Joblib

</td>
</tr>
</table>

---

# 🧠 Machine Learning Workflow

```mermaid
flowchart LR

A[Raw Dataset] --> B[Data Cleaning]
B --> C[Feature Engineering]
C --> D[Scaling & Imputation]
D --> E[Model Training]
E --> F[Hyperparameter Tuning]
F --> G[Cross Validation]
G --> H[Final XGBoost Model]
H --> I[Desktop GUI Deployment]
```

---

# 📊 Exploratory Data Analysis (EDA)

The dataset underwent a rigorous preprocessing pipeline before training.

<details>
<summary><b>🔍 Click to Expand EDA Steps</b></summary>

---

### 🧹 Data Cleaning

* Removed redundant features like:

  * `infant deaths`
  * `Country`
  * `Year`

This helped the model learn generalized healthcare patterns instead of memorizing identities.

---

### 📌 Missing Value Handling

Implemented:

```python
Median Imputation
```

to preserve dataset integrity while handling null values.

---

### 📏 Feature Scaling

Applied:

```python
StandardScaler
```

to normalize feature ranges and improve model stability.

---

### 📈 Statistical Insights

Performed:

* Correlation Analysis
* Feature Importance Analysis
* Distribution Analysis
* Comparative Benchmarking

---

</details>

---

# 🤖 Machine Learning Engine

## 🔥 Hyperparameter Tuned XGBoost

The model was optimized using carefully tuned parameters:

```python
XGBRegressor(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 6
)
```

### Why These Parameters?

| Parameter            | Purpose                                      |
| -------------------- | -------------------------------------------- |
| `n_estimators=500`   | More boosting rounds for residual correction |
| `learning_rate=0.05` | Stable convergence & reduced overfitting     |
| `max_depth=6`        | Balanced complexity and generalization       |

---

# 🛡️ Robust Validation Strategy

Instead of relying on a single train-test split, the project incorporates:

# ✅ Cross Validation (K-Fold Logic)

This ensures:

* Better generalization
* Reduced overfitting risk
* Stable performance across data subsets
* Reliable real-world deployment

---

# 📈 Model Benchmarking

Four different regression models were benchmarked.

## 🏆 Performance Comparison

| Model                   | R² Accuracy |
| ----------------------- | ----------- |
| Linear Regression       | 81.20%      |
| Poisson Regression      | 81.67%      |
| Random Forest           | 96.76%      |
| **XGBoost (Optimized)** | **97.07%**  |

---

# 📊 Visual Results

## 📌 Model Comparison

<p align="center">
<img src="model_comparison_scatter.png" width="850">
</p>

---

## 📌 Feature Importance

The model identified the strongest predictors affecting global life expectancy.

### Top Predictive Factors

* HIV/AIDS prevalence
* Income composition of resources
* Schooling
* Adult mortality
* Healthcare expenditure

<p align="center">
<img src="feature_importance.png" width="850">
</p>

---

## 📌 Actual vs Predicted Comparison

To validate production reliability, predictions were compared against real historical country data.

<p align="center">
<img src="country_sample_comparison.png" width="850">
</p>

---

# 🖥️ Desktop GUI Application

The final deliverable includes a functional desktop application built using **Tkinter**.

Users can:

* Input healthcare indicators
* Modify socio-economic metrics
* Generate real-time life expectancy predictions

---

## 📱 Application Preview

<table align="center">
<tr>
<td align="center">

### High Resource Scenario

<img src="appsnapshot1.png" width="400">

</td>

<td align="center">

### Low Resource Scenario

<img src="appsnapshot.png" width="400">

</td>
</tr>
</table>

---

# 📂 Project Structure

```bash
├── Life_Expectancy_Data.csv
├── model_training.py
├── app.py
├── life_model.pkl
├── scaler.pkl
├── medians.pkl
├── feature_names.pkl
├── model_comparison_scatter.png
├── feature_importance.png
├── country_sample_comparison.png
├── appsnapshot.png
├── appsnapshot1.png
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Life-Expectancy-AI.git
cd Life-Expectancy-AI
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Application

```bash
python app.py
```

---

# 📦 Required Libraries

```python
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
tkinter
```

---

# 🎯 Key Achievements

✅ Built complete end-to-end ML pipeline
✅ Achieved 97.07% prediction accuracy
✅ Performed hyperparameter optimization
✅ Developed deployable desktop application
✅ Implemented scalable preprocessing pipeline
✅ Conducted comparative model benchmarking
✅ Applied cross-validation for robust evaluation

---

# 🔬 Future Improvements

* Flask / FastAPI Web Deployment
* Docker Containerization
* Real-Time Cloud Prediction API
* SHAP Explainability Integration
* Streamlit Dashboard Version
* Automated Retraining Pipeline

---

# 🏆 Conclusion

This project demonstrates how advanced ensemble learning techniques like **XGBoost** combined with:

* rigorous preprocessing,
* feature engineering,
* statistical analysis,
* and hyperparameter tuning

can achieve highly accurate predictions for complex socio-economic problems.

The system is not just an experimental notebook — it is a **deployable AI product** designed for practical predictive analysis and decision support applications.

---

# 👨‍💻 Author

## Ganesh Deshmukh

**M.Tech Robotics & AI**
Passionate about:

* Artificial Intelligence
* Machine Learning
* Robotics
* Data Science
* Intelligent Automation

---

<p align="center">

### ⭐ If you found this project useful, consider giving it a star!

</p>
