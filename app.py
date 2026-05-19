import os
import sys
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import numpy as np
import xgboost # <--- CRITICAL: Keeps the EXE from crashing

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Load Assets ---
try:
    model = joblib.load(resource_path('life_model.pkl'))
    scaler = joblib.load(resource_path('scaler.pkl'))
    medians = joblib.load(resource_path('medians.pkl'))
    feature_names = joblib.load(resource_path('feature_names.pkl'))
except Exception as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Loading Error", f"Could not load AI files: {e}")
    sys.exit()

def predict_life():
    try:
        # 1. Start with the average (median) for ALL features
        user_input_dict = medians.copy()
        
        # 2. Overwrite only the ones the user typed
        # Note: These keys MUST match the CSV column names exactly
        user_input_dict['Schooling'] = float(ent_school.get())
        user_input_dict['GDP'] = float(ent_gdp.get())
        user_input_dict['BMI'] = float(ent_bmi.get())
        user_input_dict['Alcohol'] = float(ent_alcohol.get())
        user_input_dict['Adult Mortality'] = float(ent_mortality.get())

        # 3. Force columns into the EXACT order the model was trained on
        input_df = pd.DataFrame([user_input_dict])
        input_df = input_df[feature_names] # This fixes the "constant value" bug
        
        # 4. Scale and Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        # 5. Show Result
        lbl_res.config(text=f"Estimated Life Expectancy:\n{round(prediction, 2)} Years", fg="#2ecc71")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")
    except Exception as e:
        messagebox.showerror("System Error", f"Error: {e}")

# --- UI Setup ---
root = tk.Tk()
root.title("Life Expectancy Predictor v1.0")
root.geometry("400x550")
root.configure(padx=20, pady=20)

tk.Label(root, text="Life Expectancy AI", font=("Arial", 18, "bold")).pack(pady=10)

# Create input fields
def create_field(label_text):
    tk.Label(root, text=label_text, font=("Arial", 10)).pack(anchor="w")
    e = tk.Entry(root, font=("Arial", 10))
    e.pack(fill="x", pady=5)
    return e

ent_school = create_field("Schooling (Years):")
ent_gdp = create_field("GDP per Capita:")
ent_bmi = create_field("BMI:")
ent_alcohol = create_field("Alcohol Consumption:")
ent_mortality = create_field("Adult Mortality Rate:")

tk.Button(root, text="PREDICT NOW", command=predict_life, bg="#3498db", fg="white", 
          font=("Arial", 12, "bold"), height=2).pack(fill="x", pady=20)

lbl_res = tk.Label(root, text="", font=("Arial", 14, "bold"))
lbl_res.pack()

root.mainloop()