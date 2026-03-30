import pandas as pd
import pickle

# 1. Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first.")
    exit()

print("--- Student Sleep Predictor ---")
print("Enter today's activities to predict tonight's sleep.\n")

# 2. Get input from the user
study = float(input("Hours spent studying: "))
coffee = float(input("Cups of coffee/tea consumed: "))
screen = float(input("Hours of screen time (phone/laptop): "))
deadlines = float(input("Number of deadlines in next 48 hours: "))
exercise = float(input("Hours of physical exercise: "))

# 3. Format the input into a DataFrame (matches training data structure)
input_data = pd.DataFrame({
    'StudyHours': [study],
    'CaffeineCups': [coffee],
    'ScreenTimeHours': [screen],
    'Deadlines': [deadlines],
    'ExerciseHours': [exercise]
})

# 4. Make the Prediction
predicted_sleep = model.predict(input_data)[0]

print("\n--------------------------------")
print(f"Predicted Sleep: {predicted_sleep:.2f} hours")
print("--------------------------------")