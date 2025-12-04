import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # Library to save the model

# Load the training and validation sets
X_train = pd.read_csv('data/X_train.csv')
X_val = pd.read_csv('data/X_val.csv')
y_train = pd.read_csv('data/y_train.csv')
y_val = pd.read_csv('data/y_val.csv')

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train.values.ravel())  # Train the model

# Save the trained model
joblib.dump(model, 'model/logistic_regression_model.pkl')
print("Model saved as 'model/logistic_regression_model.pkl'")

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")