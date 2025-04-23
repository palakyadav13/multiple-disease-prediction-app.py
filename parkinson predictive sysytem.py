import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("C:\ML\parkinsons.csv")

# Features and labels
X = data.drop(columns=['name', 'status'], axis=1)
y = data['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = SVC()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open("C:/train/parkinsons_model.sav", "wb") as f:
    pickle.dump(model, f)

with open("C:/train/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Example prediction with new input
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,
              0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,
              1.743867,0.085569)
input_data_reshaped = np.array(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)

# Output result
if prediction[0] == 0:
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")
