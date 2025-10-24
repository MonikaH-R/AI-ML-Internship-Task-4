import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv("data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: data.csv not found. Place the file in the same directory as this script.")
    exit()

# --- Data Cleaning and Preparation ---

# Drop unnecessary columns: 'id' and 'Unnamed: 32' (mostly NaN)
df = df.drop(columns=['id', 'Unnamed: 32'])

# Encode the target variable 'diagnosis': M=1 (Malignant), B=0 (Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features (X) and target (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# --- 2. Train/Test Split and Standardize Features ---

# Split the data into training (70%) and testing (30%) sets, ensuring stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features: Scale the data for optimal Logistic Regression performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {len(X_train_scaled)} samples")
print(f"Testing set size: {len(X_test_scaled)} samples")

# --- 3. Fit a Logistic Regression Model ---

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
print("\nLogistic Regression model trained successfully.")

# Predict class labels and class probabilities
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# --- 4. Evaluate the Model (Default Threshold: 0.5) ---

print("\n" + "="*50)
print("Model Evaluation (Standard Threshold: 0.5)")
print("="*50)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)'])
print("\nClassification Report:\n", class_report)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# --- Visualization and File Saving ---

# Plot and Save Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Standard Threshold: 0.5)')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign', 'Malignant'])
plt.yticks(tick_marks, ['Benign', 'Malignant'])
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Save the plot BEFORE showing/closing
plt.savefig('confusion_matrix.png')
print("\nImage saved: confusion_matrix.png")
plt.show() # Display the plot
plt.close()

# Plot and Save ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)

# Save the plot BEFORE showing/closing
plt.savefig('roc_curve.png')
print("Image saved: roc_curve.png")
plt.show() # Display the plot
plt.close()

# --- 5. Threshold Tuning Example (Youden's J statistic) ---

# Calculate Optimal Threshold using Youden's J statistic (J = TPR - FPR)
J = tpr - fpr
ix = np.argmax(J)
optimal_threshold = thresholds[ix]

print("\n" + "="*50)
print("Threshold Tuning")
print("="*50)
print(f"Optimal Threshold (Youden's J statistic): {optimal_threshold:.4f}")

# Re-evaluate metrics with the optimal threshold for comparison
y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)
tuned_report = classification_report(y_test, y_pred_tuned, target_names=['Benign (0)', 'Malignant (1)'])
print(f"\nClassification Report with Optimal Threshold ({optimal_threshold:.4f}):\n", tuned_report)