🌍 Life Expectancy Intelligence: End-to-End ML Pipeline
![alt text](https://img.shields.io/badge/Python-3.8+-blue.svg)

![alt text](https://img.shields.io/badge/XGBoost-97.07%25-green.svg)

![alt text](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
📌 Project Overview
This project is a production-ready, end-to-end Machine Learning solution designed to predict life expectancy based on global socio-economic and health indicators. By leveraging an optimized XGBoost Regressor, the system achieves an industry-leading 97.07% accuracy (R² Score).
The project transitions from raw data cleaning and deep statistical analysis to a finalized Desktop Application for real-time predictions.
📱 Final Product: Desktop GUI
The final deliverable is a functional desktop application built with Tkinter. It allows users to input key metrics and receive instant, AI-driven life expectancy estimates.
High-Resource Scenario	Low-Resource Scenario
![alt text](appsnapshot1.png)
![alt text](appsnapshot.png)
Accurate predictions for high-GDP regions.	Smart handling of varying health metrics.
📊 1. Exploratory Data Analysis (EDA)
Before modeling, the data underwent a rigorous cleaning process:
Feature Removal: Dropped infant deaths (redundancy) and Country/Year to ensure the model learns generalized health patterns rather than specific identities.
Imputation: Used Median Imputation to handle missing values, ensuring no data was lost while maintaining statistical integrity.
Scaling: Applied StandardScaler to normalize features, ensuring that variables with large ranges (like GDP) don't overpower smaller ones (like Alcohol consumption).
🧠 2. The Machine Learning Engine
Hyperparameter Tuning
To push the model to 97.07% accuracy, I moved beyond default settings. I tuned the following parameters in XGBoost:
n_estimators=500: Allowed the model more boosting rounds to correct residual errors.
learning_rate=0.05: Used a slow learning pace to ensure better convergence and prevent overfitting.
max_depth=6: Balanced complexity to capture non-linear relationships without capturing noise.
Robust Evaluation: K-Fold Logic
To ensure these results weren't just a "lucky split," the pipeline is built on the principles of Cross-Validation. By simulating K-Fold logic, I ensured the model's high performance is consistent across different subsets of the global data, making it reliable for real-world deployment.
📈 3. Model Comparison & Benchmarking
I benchmarked four distinct algorithms to justify the selection of XGBoost. As seen below, the ensemble methods (Random Forest and XGBoost) significantly outperformed standard linear models.
![alt text](model_comparison_scatter.png)
Model	Accuracy (R²)
Linear Regression	81.20%
Poisson Regression	81.67%
Random Forest	96.76%
XGBoost (Extreme)	97.07%
🔎 4. Key Drivers & Generalization
Feature Importance
By analyzing the model's internal weights, we identified the most critical factors influencing global life expectancy. HIV/AIDS prevalence and Income composition of resources emerged as the top predictors.
![alt text](feature_importance.png)
Sample Comparison (Actual vs. Predicted)
To verify the model's reliability in production, I compared the predictions against actual historical data for 20 diverse countries. The variance is nearly indistinguishable, proving the model's high precision.
![alt text](country_sample_comparison.png)
🛠️ Requirements & Installation
Library Dependencies
The project requires the following libraries:
code
Text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
Installation & Setup
Clone the Repository:
code
Bash
git clone https://github.com/your-username/Life-Expectancy-AI.git
Install Requirements:
code
Bash
pip install -r requirements.txt
Run the Application:
code
Bash
python app.py
📂 Project Structure
code
Text
├── Life_Expectancy_Data.csv    # Source Dataset
├── model_training.py           # EDA, Tuning, and Benchmarking Script
├── app.py                      # Tkinter GUI Application
├── life_model.pkl              # Trained XGBoost Model
├── scaler.pkl                  # Fitted StandardScaler
├── medians.pkl                 # Stored medians for app imputation
├── feature_names.pkl           # Ordered list of features
└── README.md                   # Project Documentation
🏆 Conclusion
This project demonstrates that through rigorous Hyperparameter Tuning and Feature Engineering, a robust ensemble model like XGBoost can provide near-perfect predictive power for complex social metrics. This tool is not just a script, but a finalized AI product ready for decision-making analysis.