import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv('data.csv')

# 2. Separate Features (X) and the Target Variable (y)
X = df[['StudyHours', 'CaffeineCups', 'ScreenTimeHours', 'Deadlines', 'ExerciseHours']]
y = df['SleepHours']

# 3. Initialize and Train the Model (Supervised Learning)
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X, y)

# 4. Feature Analysis (Which features matter most?)
print("\n--- Feature Impact on Sleep ---")
print("(Negative values mean this activity reduces sleep)")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f} hours")
print("-------------------------------\n")

# 5. Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Success! Model saved as 'model.pkl'")