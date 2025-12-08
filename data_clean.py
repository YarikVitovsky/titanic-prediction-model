import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/train.csv')
scaler = StandardScaler()

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'])
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'] = data['Age'].fillna(data['Age'].mean())

missing_values = data.isnull().sum()
print(missing_values)

target = data['Survived']
features = data.drop(columns=['Survived'])

scaled_features = scaler.fit_transform(features)#NumPy Object
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)#converting back to data frame

data = pd.concat([scaled_features, target], axis=1)

data.to_csv('data/train_cleaned.csv', index=False)