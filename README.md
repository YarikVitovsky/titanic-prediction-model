# The Titanic Disaster: When Data Tells a Story 🚢

> *"The sea was like glass. There wasn't even a ripple on the surface of the water."* — Eva Hart, Titanic Survivor

This project goes beyond traditional machine learning to explore the human stories embedded within the Titanic dataset. Using logistic regression and data visualization, we uncover patterns of survival that reveal both the nobility and tragedy of human nature during one of history's most devastating maritime disasters.

**🎯 Key Achievement**: 79% prediction accuracy using logistic regression  
**📊 Project Report**: LaTeX document analyzing survival patterns through both mathematical and human perspectives  
**🎨 Data Visualizations**: matplotlib charts that tell the story behind the statistics

🚀 **Check out the live demo**: [https://titanic-survival-predictor-1.onrender.com/](https://titanic-survival-predictor-1.onrender.com/)

📖 **Read the Project Report**: [Project-Report.pdf](https://github.com/YarikVitovsky/titanic-dataset/blob/main/Project-Report.pdf)

---

## 📖 Project Highlights

### **📊 Data Visualization & Storytelling**
- **Interactive Charts**: Survival rates by gender, class, age, and family size
- **Human Stories**: Individual tales of heroism, sacrifice, and the "women and children first" protocol
- **Statistical Insights**: How social class, family bonds, and demographics influenced survival

### **🔬 Mathematical Foundation**
- **Logistic Regression Deep Dive**: From linear regression to sigmoid curves
- **Feature Engineering**: Family size, social titles, and economic indicators  
- **Model Interpretation**: Understanding what the algorithm learned about human behavior

### **📝 Comprehensive Report**
A beautifully crafted LaTeX report that weaves together:
- Individual stories of heroes like Captain Edward Smith and Wallace Hartley's band
- Mathematical explanations of logistic regression and the sigmoid function
- Data visualizations revealing survival patterns
- Ethical considerations about bias in historical data
- Modern applications of survival analysis

---

## 📂 Project Structure

```
├── data/
│   ├── train.csv              # Original Titanic dataset
│   ├── test.csv               # Test dataset  
│   ├── train_cleaned.csv      # Cleaned and processed data
│   └── gender_submission.csv  # Sample submission format
├── images/
│   ├── sigmoid_curve.png      # Sigmoid function visualization
│   ├── gender_survival.png    # Survival rates by gender
│   ├── class_survival.png     # Survival rates by class
│   ├── age_survival.png       # Age distribution and survival
│   └── family_survival.png    # Family size impact on survival
├── model/
│   └── logistic_regression_model.pkl  # Trained model
├── data_clean.py              # Data cleaning and preprocessing
├── data_split.py              # Train/validation split
├── model_train.py             # Logistic regression training
├── generate_visualizations.py # Creates all charts and graphs
└── report.tex                 # LaTeX report source (6 pages)
```

### **🛠 Core Components**

1. **Data Cleaning & Feature Engineering**
   - Missing value imputation using statistical methods
   - Feature creation: `FamilySize`, `IsAlone`, title extraction
   - Categorical encoding and data normalization

2. **Visualization Engine** 
   - Matplotlib/Seaborn charts showing survival patterns
   - Gender, class, age, and family dynamics analysis
   - Statistical storytelling through data

3. **Machine Learning Pipeline**
   - Logistic regression with 79% validation accuracy
   - Sigmoid function implementation and interpretation
   - Feature importance analysis

4. **Comprehensive Documentation**
   - LaTeX report combining technical rigor with human narrative
   - Mathematical foundations explained intuitively  
   - Ethical implications and modern applications

---

## 🚀 How to Run the Project

### **Prerequisites**
- Python 3.8+ installed
- Git for cloning the repository

### **1. Clone the Repository**
```bash
git clone https://github.com/YarikVitovsky/titanic-dataset.git
cd titanic-dataset
```

### **2. Set Up the Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn joblib
```

### **3. Run the Complete Pipeline**

1. **Data Cleaning & Processing**:
   ```bash
   python data_clean.py
   ```
   *Outputs*: `data/train_cleaned.csv` with processed features

2. **Create Train/Validation Split**:
   ```bash
   python data_split.py
   ```
   *Outputs*: `X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`

3. **Train the Model**:
   ```bash
   python model_train.py
   ```
   *Outputs*: `model/logistic_regression_model.pkl` with 79% accuracy

4. **Generate Visualizations**:
   ```bash
   python generate_visualizations.py
   ```
   *Outputs*: All charts saved to `images/` folder

5. **Compile the Report** (optional):
   ```bash
   # If you have LaTeX installed
   pdflatex report.tex
   ```

---

## 📊 Key Insights & Results

### **Survival Statistics**
- **Overall Survival Rate**: 38.4% (342 of 891 passengers)
- **Women**: 74.2% survival rate → "Women first" protocol in action
- **Men**: 18.9% survival rate → Ultimate sacrifice for others
- **Children**: 52% survival rate → "Children first" principle
- **First Class**: 63% survival rate vs **Third Class**: 24%

### **Human Stories Revealed by Data**
- **Family Bonds**: Optimal family size (2-4 members) showed highest survival rates
- **Social Hierarchies**: Passenger titles (`Mr.`, `Mrs.`, `Miss`) revealed class distinctions  
- **Economic Reality**: Higher fares directly correlated with better survival chances
- **Individual Heroes**: Captain Smith, Wallace Hartley's band, Ida & Isidor Straus

### **Mathematical Beauty**
The sigmoid function S(z) = 1/(1+e^(-z)) transforms passenger features into survival probabilities:
- **z < -2**: Very low survival probability (mostly third-class men)
- **z ≈ 0**: 50% probability (borderline cases where details matter)
- **z > 2**: Very high survival probability (first-class women/children)

---

## 🎨 Visualizations

The project generates several compelling visualizations:

- **`gender_survival.png`**: Dramatic difference between male/female survival rates
- **`class_survival.png`**: Economic inequality's deadly impact  
- **`age_survival.png`**: How age influenced survival chances
- **`family_survival.png`**: Sweet spot of family size for survival
- **`sigmoid_curve.png`**: Mathematical foundation of logistic regression

---

## 📋 Technical Features

### **Data Science Pipeline**
- **Missing Value Handling**: Statistical imputation methods
- **Feature Engineering**: Created `FamilySize`, `IsAlone`, title extraction
- **Data Normalization**: Scaled features for optimal model performance
- **Train/Validation Split**: Proper evaluation methodology

### **Machine Learning**
- **Algorithm**: Logistic Regression with sigmoid activation
- **Accuracy**: 79% on validation set
- **Interpretability**: Clear coefficient analysis showing feature importance
- **Ethical Awareness**: Discussion of historical bias in data

### **Visualization & Reporting**
- **Matplotlib/Seaborn**: Professional statistical charts
- **LaTeX Report**: 6-page comprehensive analysis
- **Storytelling**: Human narrative woven through technical analysis

---

## 🛠 Technologies Used

- **Python**: Core data processing and analysis
- **Scikit-learn**: Logistic regression implementation  
- **Pandas**: Data manipulation and cleaning
- **Matplotlib & Seaborn**: Statistical visualization and charts
- **LaTeX**: Professional report generation
- **Joblib**: Model serialization and saving
- **NumPy**: Numerical computations

---

## 📚 What Makes This Project Special

### **Beyond Standard ML Projects**
This isn't just another machine learning exercise. It's a thoughtful exploration of how data science can illuminate human stories while maintaining respect for historical tragedy.

### **Educational Value**
- **Intuitive Math Explanations**: Logistic regression explained as naturally as linear regression
- **Human Context**: Every statistic connected to real human experiences  
- **Ethical Considerations**: Discussion of bias, representation, and algorithmic responsibility

### **Professional Presentation**
- **Academic-Quality Report**: LaTeX document suitable for research or portfolio
- **Beautiful Visualizations**: Publication-ready charts and graphs
- **Complete Documentation**: From code comments to comprehensive README

---

## 📄 Dataset & Sources

- **Primary Dataset**: [Titanic Kaggle Competition](https://www.kaggle.com/c/titanic)
- **Historical Context**: Survivor accounts and maritime records
- **Ethical Framework**: Treating data subjects with dignity and respect

---

## 🏆 Learning Outcomes

After exploring this project, you'll understand:
- **Logistic Regression**: From mathematical foundation to practical implementation
- **Feature Engineering**: Creating meaningful variables from raw data  
- **Data Storytelling**: Using visualization to reveal human narratives
- **Ethical AI**: Considering bias, representation, and historical context
- **Professional Documentation**: Creating reports that combine rigor with accessibility

---

## 🤝 Contributing

We welcome contributions that enhance the project's educational value or improve the analysis:

- **Historical Insights**: Additional passenger stories or historical context
- **Visualization Ideas**: New ways to present the data
- **Code Improvements**: Better documentation or optimization
- **Report Enhancements**: Expanding the LaTeX document

Feel free to fork and submit pull requests!

---

## 📧 Contact & Links

**Yarik Vitovsky**
- **Email**: yarikvitovsky@gmail.com
- **GitHub**: [https://github.com/YarikVitovsky](https://github.com/YarikVitovsky)
- **Live Demo**: [Titanic Survival Predictor](https://titanic-survival-predictor-1.onrender.com/)

---

## 💭 Final Reflection

*"Behind every dataset lies human experience. The true measure of our models is not just their accuracy, but their humanity."*

This project honors the memory of the 1,514 souls lost in the North Atlantic while demonstrating how thoughtful data analysis can preserve and illuminate their stories for future generations.

### 🚢 "Women and children first!" - A principle that shaped survival rates on the Titanic.