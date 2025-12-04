# Titanic Survival Prediction Project 🚢

This project uses machine learning to predict whether a passenger would have survived the Titanic disaster based on various features such as age, gender, ticket class, and more. The dataset is sourced from the famous Titanic Kaggle competition.

---

## 📂 Project Structure

### **1. Data Cleaning**
- Removed irrelevant columns (e.g., `Cabin`).
- Handled missing values:
  - Numeric columns filled with the mean.
  - Categorical columns filled with the mode.
- Encoded categorical variables (`Sex`, `Embarked`).

### **2. Feature Engineering**
- Created new features:
  - `FamilySize`: Total number of family members aboard.
  - `IsAlone`: Binary feature indicating if the passenger was traveling alone.
- Extracted and encoded titles (e.g., `Mr`, `Mrs`, `Miss`) from the `Name` column.

### **3. Data Splitting**
- Split the dataset into training and validation sets:
  - `X_train.csv`: Training features.
  - `X_val.csv`: Validation features.
  - `y_train.csv`: Training target.
  - `y_val.csv`: Validation target.

### **4. Model Training**
- Trained a **Logistic Regression** model using the training set.
- Achieved a validation accuracy of **79%**.
- Saved the trained model as `logistic_regression_model.pkl`.

---

## 🚀 How to Run the Project

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd Titanic-Dataset
```

### **2. Set Up the Environment**
- Install Python dependencies:
```bash
pip install -r requirements.txt
```

### **3. Run the Scripts**
1. **Data Cleaning**:
   ```bash
   python data_clean.py
   ```
   - Outputs: `train_cleaned.csv`

2. **Data Splitting**:
   ```bash
   python data_split.py
   ```
   - Outputs: `X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`

3. **Model Training**:
   ```bash
   python model_train.py
   ```
   - Outputs: `logistic_regression_model.pkl`

---

## 🌟 Features

- **Data Cleaning**: Handles missing values and irrelevant columns.
- **Feature Engineering**: Adds meaningful features like `FamilySize` and `IsAlone`.
- **Model Training**: Trains a Logistic Regression model to predict survival.
- **Model Saving**: Saves the trained model for future use.

---

## 📊 Results

- **Validation Accuracy**: **79%**
- The model predicts survival based on features like:
  - Ticket class (`Pclass`)
  - Gender (`Sex`)
  - Age
  - Number of siblings/spouses aboard (`SibSp`)
  - Number of parents/children aboard (`Parch`)
  - Fare
  - Port of embarkation (`Embarked`)
  - Family size (`FamilySize`)
  - Whether the passenger was alone (`IsAlone`)
  - Title (`Mr`, `Mrs`, etc.)

---

## 🛠 Technologies Used

- **Python**: Data processing and model training.
- **Scikit-learn**: Machine learning library.
- **Pandas**: Data manipulation.
- **Joblib**: Model saving.
- **Flask/React/Node.js** (planned): For building a web application.

---

## 📄 Dataset

The dataset is sourced from the [Titanic Kaggle Competition](https://www.kaggle.com/c/titanic).

---

## 🤝 Contributing

Feel free to fork the repository and submit pull requests for improvements or new features!

---

## 📧 Contact

For questions or collaborations, reach out at:
- **Email**: yarikvitovsky@gmail.com
- **GitHub**: [https://github.com/YarikVitovsky](https://github.com/YarikVitovsky)

---

### 🚢 "Women and children first!" - A principle that shaped survival rates on the Titanic.