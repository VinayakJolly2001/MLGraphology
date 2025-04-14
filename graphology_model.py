#rule based graphology
import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def calculate_slant(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image not found!", "N/A"

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray_image, (700, 700), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(resized_gray, 50, 150)                        #detect edges of letter or words
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)    #detecting straight lines in your image

    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]        # each line is represented by [[rho, theta]]
            angle = np.rad2deg(theta)     #This converts the angle from radians to degrees.
            if angle > 90:
                angle -= 180
            if -45 <= angle <= 45:
                angles.append(angle)
    
    avg_slant = round(float(np.mean(angles)), 2) if angles else 0.0
    
    if avg_slant > 0:
        return avg_slant, "Sociable, open, emotionally expressive"
    else:
        return avg_slant, "Introverted, reserved, emotionally controlled"

def calculate_letter_size_spacing(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, 0.0, "N/A"

    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)       #This is done so text becomes white (foreground) and background becomes black, which helps in contour detection.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #This finds the outlines of the white shapes

    sizes = []
    spaces = []
    prev_x = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        sizes.append(w * h)
        if prev_x is not None:
            spaces.append(abs(x - prev_x))          #Calculates the gap between current letter and the previous one by subtracting their starting x positions.
        prev_x = x + w                              #Updates the end of the current letter (right edge), to be used for spacing with the next letter.


    avg_size = round(float(np.mean(sizes)), 2) if sizes else 0.0
    avg_spacing = round(float(np.mean(spaces)), 2) if spaces else 0.0
    
    size_trait = "Outgoing, extroverted, attention-seeking" if avg_size > 500 else "Focused, meticulous, introverted"
    return avg_size, avg_spacing, size_trait

def calculate_loop_size(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "N/A"

    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)    # text becomes white on black background,
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #This function detects contours, which are basically outlines or borders of white regions
    loop_areas = []
    
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:  # Ensuring it's an inner loop
                area = cv2.contourArea(contours[i])
                loop_areas.append(area)

    avg_loop_size = round(float(np.mean(loop_areas)), 2) if loop_areas else 0.0

    # Updated graphology-based conditions
    if avg_loop_size < 300:
        return "Self-conscious, avoids expressing emotions"
    elif 300 <= avg_loop_size <= 700:
        return "Balanced, expressive but thoughtful"
    else:
        return "Highly sensitive, creative, open"

def analyze_handwriting(image_path):
    slant_angle, slant_trait = calculate_slant(image_path)
    avg_size, avg_spacing, size_trait = calculate_letter_size_spacing(image_path)
    loop_trait = calculate_loop_size(image_path)
    
    return {
        "Slant Angle": slant_angle,
        "Personality (Slant)": slant_trait,
        "Avg Letter Size": avg_size,
        "Personality (Letter Size)": size_trait,
        "Loop Size": loop_trait
    }

def save_to_excel(data, filename="handwriting_analysis2.xlsx"):
    df = pd.DataFrame([data])
    
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(filename, index=False)
    print(f"Data successfully saved to {filename}")

def upload_and_analyze():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        results = analyze_handwriting(file_path)
        save_to_excel(results)
        for key, value in results.items():
            print(f"{key}: {value}")

# Run the upload function
upload_and_analyze()
