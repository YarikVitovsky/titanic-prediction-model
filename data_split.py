import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv('data/train_cleaned.csv')

# Define features (X) and target (y)
X = df.drop('Survived', axis=1)  # Features: all columns except 'Survived'
y = df['Survived']  # Target: 'Survived'

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and validation sets to CSV files
X_train.to_csv('data/X_train.csv', index=False)
X_val.to_csv('data/X_val.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_val.to_csv('data/y_val.csv', index=False)

print("Data split complete. Training and validation sets saved.")