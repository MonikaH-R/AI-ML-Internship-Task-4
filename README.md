# AI-ML-Internship-Task-4


###  **Objective**

The goal of this task is to **build a binary classification model** using **Logistic Regression**. The model predicts
whether a tumor is **Malignant (1)** or **Benign (0)** based on the Breast Cancer dataset.


###  **Tools & Libraries Used**

* **Python 3**
* **Pandas** – for data handling and preprocessing
* **NumPy** – for numerical operations
* **Scikit-learn** – for model building, evaluation, and metrics
* **Matplotlib** – for data visualization

###  **Dataset**

* File: `data.csv`
* Source: Breast Cancer Wisconsin (Diagnostic) dataset
* Target column: `diagnosis`

  * **M = 1 (Malignant)**
  * **B = 0 (Benign)**

Unnecessary columns such as `id` and `Unnamed: 32` were dropped before training.


###  **Steps Followed**

#### 1️ Load and Prepare Data

* Read the dataset using `pandas`.
* Removed unwanted columns.
* Encoded the target variable (`M` → 1, `B` → 0).
* Split the data into **features (X)** and **labels (y)**.

#### 2️ Train/Test Split & Feature Scaling

* Split dataset: **70% training**, **30% testing** using `train_test_split` with stratification.
* Standardized all features using `StandardScaler` to improve model performance.

#### 3️ Model Training

* Trained a **Logistic Regression model** using `sklearn.linear_model.LogisticRegression`.
* Predicted both **class labels** and **class probabilities**.

#### 4️ Model Evaluation

Evaluated model performance using:

* **Confusion Matrix**
* **Precision, Recall, F1-Score (Classification Report)**
* **ROC-AUC Score**
* **ROC Curve Visualization**

Generated and saved:

* 📊 `confusion_matrix.png`
* 📈 `roc_curve.png`

#### 5️ Threshold Tuning (Youden’s J Statistic)

* Computed optimal threshold to balance **True Positive Rate (TPR)** and **False Positive Rate (FPR)**.
* Re-evaluated model using the new threshold for improved performance.


###  **Key Metrics (Example Output)**

| Metric                   | Description                               |
| :----------------------- | :---------------------------------------- |
| **Precision**            | How accurate the positive predictions are |
| **Recall (Sensitivity)** | How well the model detects positives      |
| **F1-Score**             | Balance between Precision and Recall      |
| **ROC-AUC**              | Overall model classification performance  |


####  Confusion Matrix

Displays the number of true/false positives and negatives to understand model prediction accuracy.

####  ROC Curve

Shows the trade-off between **sensitivity (recall)** and **specificity (1 - false positive rate)**.
A higher **AUC** indicates a stronger model.


###  **Sigmoid Function Explanation**

Logistic Regression uses the **sigmoid (logistic) function** to map predictions to probabilities between 0 and 1:
[
\sigma(x) = \frac{1}{1 + e^{-x}}
]

* If probability ≥ threshold → Predict **1 (Malignant)**
* If probability < threshold → Predict **0 (Benign)**



###  **Results Summary**

* The model effectively classifies malignant and benign tumors.
* After threshold tuning, **recall improved**, indicating better detection of malignant cases.
* ROC-AUC score close to **1.0** shows excellent performance.


###  **Project Files**

| File Name                      | Description                    |
| ------------------------------ | ------------------------------ |
| `logistic_regression_task4.py` | Main Python script             |
| `data.csv`                     | Input dataset                  |
| `confusion_matrix.png`         | Confusion matrix visualization |
| `roc_curve.png`                | ROC curve visualization        |
| `README.md`                    | Project documentation          |



### **Conclusion**

This project demonstrates how **Logistic Regression** can be used for **binary classification problems**, including:

* Feature scaling
* Model training
* Evaluation using standard metrics
* Threshold tuning for better performance
