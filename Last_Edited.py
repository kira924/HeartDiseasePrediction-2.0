import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score)

# --- Load Data for Regression ---
heart_data_linear_regression = pd.read_csv(r"D:\study\.vscode\AI_PL\project\heartRisk.csv")
X_regression = heart_data_linear_regression.drop(columns="risk", axis=1)
Y_regression = heart_data_linear_regression["risk"]
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X_regression, Y_regression, test_size=0.2 , random_state= 42)

# Train Regression Model
poly = PolynomialFeatures(degree=3)  
X_poly_train = poly.fit_transform(X_train_reg)
X_poly_test = poly.fit_transform(X_test_reg)

model_regression = Ridge(alpha=1) 
model_regression.fit(X_poly_train, Y_train_reg)

# --- Load Data for Classification ---
heart_data_classification = pd.read_csv(r"D:\study\.vscode\AI_PL\project\heart_disease_data.csv")
X_classification = heart_data_classification.drop(columns="target", axis=1)
Y_classification = heart_data_classification["target"]
X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(X_classification,Y_classification,test_size=0.2,random_state=2)

# Train Classification Model
model_classification = LogisticRegression()
model_classification.fit(X_train_class, Y_train_class)

def show_prediction_popup(result_text):
    """Create a popup window to show the prediction result."""
    popup = tk.Toplevel()
    popup.title("Prediction Result")
    popup.geometry("600x200")
    popup.configure(bg="#FFEBEE")

    result_label = tk.Label(popup, text=result_text, font=("Arial", 16, "bold"), bg="#FFEBEE", fg="#B71C1C")
    result_label.pack(pady=50)

    close_button = tk.Button(
        popup,
        text="Close",
        command=popup.destroy,
        bg="#D32F2F",
        fg="white",
        font=("Arial", 12, "bold"),
    )
    close_button.pack()

def classification_window(frame):
    title_label = tk.Label(frame, text="Heart Disease Classification", font=("Arial", 20, "bold"), bg="#FFEBEE", fg="#B71C1C")
    title_label.grid(row=0, column=0, columnspan=2, pady=10)
    
    age_entry = tk.Entry(frame)
    sex_var = tk.Entry(frame)
    cp_var = tk.Entry(frame)
    trestbps_entry = tk.Entry(frame)
    chol_entry = tk.Entry(frame)
    fbs_var = tk.Entry(frame)
    restecg_var = tk.Entry(frame)
    thalach_entry = tk.Entry(frame)
    exang_var = tk.Entry(frame)
    oldpeak_entry = tk.Entry(frame)
    slope_var = tk.Entry(frame)
    ca_entry = tk.Entry(frame)
    thal_var = tk.Entry(frame)
    
    fields = [
        ("Age", age_entry),
        ("Sex (0=Female, 1=Male)", sex_var),
        ("CP Type (0-3)", cp_var),
        ("Trestbps", trestbps_entry),
        ("Cholesterol", chol_entry),
        ("FBS (0,1)", fbs_var),
        ("Rest ECG (0-2)", restecg_var),
        ("Max Heart Rate (Thalach)", thalach_entry),
        ("Exang (0,1)", exang_var),
        ("Oldpeak", oldpeak_entry),
        ("Slope (0-2)", slope_var),
        ("Number of Major Vessels (CA)", ca_entry),
        ("Thalassemia (1-3)", thal_var),
    ]
    
    row = 1
    for label, widget in fields:
        label_widget = tk.Label(frame, text=label, font=("Arial", 12), bg="#FFEBEE")
        label_widget.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        widget.grid(row=row, column=1, padx=10, pady=5)
        row += 1


    def load_data_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV and Excel files", "*.csv;*.xlsx;*.xls")])
        if file_path:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:
                    data = pd.read_excel(file_path)
                select_sample(data, fields)


    def select_sample(data, fields):
        selection_window = tk.Toplevel()
        selection_window.title("Select a Sample")
        selection_window.geometry("600x400")
        
        listbox = tk.Listbox(selection_window, width=80, height=20)
        listbox.pack(pady=10)

        for i, row in data.iterrows():
            listbox.insert(tk.END, f"Sample {i+1}: {row.values}")

        def load_sample():
            selected_index = listbox.curselection()
            if selected_index:
                row_values = data.iloc[selected_index[0]].values
                for (label, widget), value in zip(fields, row_values):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(value))
                selection_window.destroy()
        
        load_button = tk.Button(selection_window, text="Load Sample", command=load_sample)
        load_button.pack(pady=10)

    upload_button = tk.Button(frame, text="Upload Data File", command=load_data_file, bg="#FFCDD2", fg="#B71C1C", font=("Arial", 12, "bold"))
    upload_button.grid(row=row, column=0, columnspan=2, pady=10)
    

    def predict_heart_disease_logistic():
        """Predict heart disease using Logistic Regression."""
        try:
            input_data = [
                float(age_entry.get()),
                float(sex_var.get()),
                float(cp_var.get()),
                float(trestbps_entry.get()),
                float(chol_entry.get()),
                float(fbs_var.get()),
                float(restecg_var.get()),
                float(thalach_entry.get()),
                float(exang_var.get()),
                float(oldpeak_entry.get()),
                float(slope_var.get()),
                float(ca_entry.get()),
                float(thal_var.get()),]
             
            input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
            prediction = model_classification.predict(input_data_as_numpy_array)

            result = ("has Heart Disease"
                if prediction[0] == 1
                else "does NOT have Heart Disease")
            show_prediction_popup(f"Prediction: Person {result}")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid inputs for all fields.")

    predict_button_classification = tk.Button(
        frame,
        text="Predict Heart Disease",
        command=predict_heart_disease_logistic,
        bg="#D32F2F",
        fg="white",
        font=("Arial", 14),)
    predict_button_classification.grid(row=row +1, column=0, columnspan=2, pady=20)



def regression_window(frame):
    title_label = tk.Label(frame, text="Heart Risk Regression", font=("Arial", 20, "bold"), bg="#FFEBEE", fg="#B71C1C")
    title_label.grid(row=0, column=0, columnspan=2, pady=10)

    # Input fields for manual entry
    isMale_var = tk.Entry(frame)
    isSmoker_var = tk.Entry(frame)
    isDiabetic_var = tk.Entry(frame)
    isHypertensive_var = tk.Entry(frame)
    Age_entry = tk.Entry(frame)
    Systolic_entry = tk.Entry(frame)
    Cholesterol_entry = tk.Entry(frame)
    HDL_entry = tk.Entry(frame)
  

    fields = [
        ("Sex (0=Female, 1=Male)", isMale_var),
        ("Is Smoker", isSmoker_var),
        ("Is Diabetic", isDiabetic_var),
        ("Is Hypertensive", isHypertensive_var),
        ("Age", Age_entry),
        ("Systolic", Systolic_entry),
        ("Cholesterol", Cholesterol_entry),
        ("HDL", HDL_entry),
    ]

    row = 1
    for label, widget in fields:
        label_widget = tk.Label(frame, text=label, font=("Arial", 12), bg="#FFEBEE")
        label_widget.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        widget.grid(row=row, column=1, padx=10, pady=5)
        row += 1

    def load_data_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV and Excel files", "*.csv;*.xlsx;*.xls")])
        if file_path:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:
                    data = pd.read_excel(file_path)
                select_sample(data, fields)

    def select_sample(data, fields):
        selection_window = tk.Toplevel()
        selection_window.title("Select a Sample")
        selection_window.geometry("600x400")
        
        listbox = tk.Listbox(selection_window, width=80, height=20)
        listbox.pack(pady=10)

        for i, row in data.iterrows():
            listbox.insert(tk.END, f"Sample {i+1}: {row.values}")

        def load_sample():
            selected_index = listbox.curselection()
            if selected_index:
                row_values = data.iloc[selected_index[0]].values
                for (label, widget), value in zip(fields, row_values):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(value))
                selection_window.destroy()
        
        load_button = tk.Button(selection_window, text="Load Sample", command=load_sample)
        load_button.pack(pady=10)

    def predict_heart_risk():
        """Predict heart risk using Linear Regression."""
        try:
            input_data = [
                float(isMale_var.get()),
                float(isSmoker_var.get()),
                float(isDiabetic_var.get()),
                float(isHypertensive_var.get()),
                float(Age_entry.get()),
                float(Systolic_entry.get()),
                float(Cholesterol_entry.get()),
                float(HDL_entry.get()),
            ]
            input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
            input_data_as_numpy_array_poly = poly.fit_transform(input_data_as_numpy_array)
            prediction = model_regression.predict(input_data_as_numpy_array_poly)

            show_prediction_popup(f"Prediction: Heart Risk is {prediction[0]:.1f}%")
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all inputs are valid.")


    upload_button = tk.Button(frame, text="Upload Data File", command=load_data_file, bg="#FFCDD2", fg="#B71C1C", font=("Arial", 12, "bold"))
    upload_button.grid(row=row, column=0, columnspan=2, pady=10)

    predict_button = tk.Button(
        frame, 
        text="Predict Risk Score", 
        bg="#D32F2F", 
        fg="white", 
        font=("Arial", 12, "bold"),
        command=predict_heart_risk
    )
    predict_button.grid(row=row + 1, column=0, columnspan=2, pady=10)

root = tk.Tk()
root.title("Heart Disease Prediction ML Project")
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

classification_tab = tk.Frame(notebook, bg="#FFEBEE")
classification_window(classification_tab)
notebook.add(classification_tab, text="Classification")

regression_tab = tk.Frame(notebook, bg="#FFEBEE")
regression_window(regression_tab)
notebook.add(regression_tab, text="Regression")

root.mainloop()


# Model Evaluation Metrics classfication
ConfusionMatrixDisplay.from_estimator(model_classification, X_test_class, Y_test_class, cmap="Blues")

y_pred_class = model_classification.predict(X_test_class)
cm = confusion_matrix(Y_test_class, y_pred_class)

accuracy_class = accuracy_score(Y_test_class, y_pred_class)
precision_class = precision_score(Y_test_class, y_pred_class)
recall_class = recall_score(Y_test_class, y_pred_class)
f1_class = f1_score(Y_test_class, y_pred_class)

print("\nModel Evaluation Metrics(classfication):")
print("Accuracy:", round(accuracy_class * 100, 2), "%")
print("Precision:", round(precision_class * 100, 2), "%")
print("Recall:", round(recall_class * 100, 2), "%")
print("F1-Score:", round(f1_class * 100, 2), "%")

plt.title("Confusion Matrix")
plt.show()

# Model Evaluation Metrics regression
y_pred_reg = model_regression.predict(X_poly_test)

mae = mean_absolute_error(Y_test_reg, y_pred_reg)
mse = mean_squared_error(Y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test_reg, y_pred_reg)

print("\nModel Evaluation Metrics (regression):")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 2))
