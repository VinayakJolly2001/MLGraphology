import pandas as pd
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
file_path = "handwriting_analysis2.xlsx"
df = pd.read_excel(file_path)

# Encode categorical labels
le_slant = LabelEncoder()        #It is used to convert categorical (text) labels into numeric values, 
df['Personality (Slant)'] = le_slant.fit_transform(df['Personality (Slant)'])
le_size = LabelEncoder()
df['Personality (Letter Size)'] = le_size.fit_transform(df['Personality (Letter Size)'])

# Define features and targets
X = df[['Slant Angle', 'Avg Letter Size']]
y = df[['Personality (Slant)', 'Personality (Letter Size)']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')


# *Function to plot Confusion Matrix*
def plot_confusion_matrix(y_test, y_pred, le_slant, le_size):
    plt.figure(figsize=(12, 5))

    for i, col in enumerate(y_test.columns):
        cm = confusion_matrix(y_test[col], y_pred[:, i])
        plt.subplot(1, 2, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=(le_slant.classes_ if col == 'Personality (Slant)' else le_size.classes_),
                    yticklabels=(le_slant.classes_ if col == 'Personality (Slant)' else le_size.classes_))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {col}")

    plt.tight_layout()
    plt.show()
#  it builds a confusion matrix and displays it as a heatmap for easy understanding

# *Show Confusion Matrix after model training*
plot_confusion_matrix(y_test, y_pred, le_slant, le_size)


# Function to process handwriting sample
def process_handwriting(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image")
        return None
    # Dummy feature extraction
    slant_angle = np.random.uniform(-30, 30)  # Replace with actual slant calculation
    avg_letter_size = np.random.uniform(10, 50)  # Replace with actual size calc
    return [[slant_angle, avg_letter_size]]


# Function to upload image and predict personality
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    features = process_handwriting(file_path)
    if features is None:
        return

    prediction = model.predict(features)[0]
    slant_personality = le_slant.inverse_transform([prediction[0]])[0]
    size_personality = le_size.inverse_transform([prediction[1]])[0]

    result_label.config(text=f"Predicted Traits:\nSlant: {slant_personality}\nLetter Size: {size_personality}")


# GUI Setup
root = tk.Tk()
root.title("Handwriting Analysis")
root.geometry("600x400")

upload_button = tk.Button(root, text="Upload Handwriting Sample", font=("Verdana", 12), command=upload_and_predict)
upload_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Verdana", 12))
result_label.pack(pady=20)
root.mainloop()