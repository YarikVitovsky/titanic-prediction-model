import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Titanic dataset
df = pd.read_csv('data/train_cleaned.csv')  # Replace with your cleaned dataset file

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Survival Rates by Gender
def plot_gender_survival(df):
    # Create gender labels based on normalized values
    df_copy = df.copy()
    df_copy['Gender'] = df_copy['Sex'].apply(lambda x: 'Female' if x > 0 else 'Male')
    
    gender_survival = df_copy.groupby('Gender')['Survived'].mean()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=gender_survival.index, y=gender_survival.values, palette=['lightblue', 'pink'])
    plt.title('Survival Rates by Gender', fontsize=16)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.xlabel('Gender', fontsize=12)
    plt.ylim(0, 1)
    for i, v in enumerate(gender_survival.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/gender_survival.png')
    plt.close()

# 2. Survival Rates by Class
def plot_class_survival(df):
    # Create class labels based on normalized values
    df_copy = df.copy()
    def map_class(x):
        if x < -1:  # Most negative values = 1st class (best)
            return '1st Class'
        elif x > 0.5:  # Most positive values = 3rd class (worst)
            return '3rd Class' 
        else:  # Middle values = 2nd class
            return '2nd Class'
    
    df_copy['Class'] = df_copy['Pclass'].apply(map_class)
    
    class_survival = df_copy.groupby('Class')['Survived'].mean()
    # Reorder to logical sequence
    class_order = ['1st Class', '2nd Class', '3rd Class']
    class_survival = class_survival.reindex([c for c in class_order if c in class_survival.index])
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_survival.index, y=class_survival.values, palette=['gold', 'silver', 'brown'])
    plt.title('Survival Rates by Passenger Class', fontsize=16)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylim(0, 1)
    for i, v in enumerate(class_survival.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/class_survival.png')
    plt.close()

# 3. Age Distribution and Survival
def plot_age_survival(df):
    # Convert normalized age back to approximate real ages
    # Assuming the normalization was: (age - mean) / std
    # We'll create age groups based on normalized values
    df_copy = df.copy()
    
    # Map normalized age values to approximate age ranges
    def map_age_group(norm_age):
        if norm_age < -1.5:
            return "0-15 (Children)"
        elif norm_age < -0.5:
            return "16-25 (Young Adults)"
        elif norm_age < 0.5:
            return "26-35 (Adults)"
        elif norm_age < 1.5:
            return "36-50 (Middle Age)"
        else:
            return "51+ (Elderly)"
    
    df_copy['AgeGroup'] = df_copy['Age'].apply(map_age_group)
    
    # Create survival analysis by age groups
    age_survival = df_copy.groupby(['AgeGroup', 'Survived']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    age_survival.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], alpha=0.8)
    plt.title('Age Distribution and Survival Rates', fontsize=16)
    plt.xlabel('Age Groups', fontsize=12)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.legend(['Did Not Survive', 'Survived'], title='Survival')
    plt.tight_layout()
    plt.savefig('images/age_survival.png')
    plt.close()

# 4. Family Size and Survival
def plot_family_survival(df):
    # Load the original training data to get unscaled FamilySize
    original_df = pd.read_csv('data/train.csv')
    df_copy = df.copy()
    
    # Add original FamilySize to the cleaned dataframe
    df_copy['FamilySize'] = original_df['SibSp'] + original_df['Parch'] + 1

    def map_family_size(size):
        if size == 1:
            return "1 (Alone)"
        elif size <= 3:
            return "2-3 (Small Family)"
        elif size <= 5:
            return "4-5 (Medium Family)"
        else:
            return "6+ (Large Family)"
    
    df_copy['FamilySizeGroup'] = df_copy['FamilySize'].apply(map_family_size)
    
    # Group by family size group and calculate survival stats
    family_stats = df_copy.groupby('FamilySizeGroup').agg({
        'Survived': ['mean', 'count']
    }).round(3)
    
    # Flatten column names
    family_stats.columns = ['survival_rate', 'count']
    
    # Reorder categories logically
    category_order = ["1 (Alone)", "2-3 (Small Family)", "4-5 (Medium Family)", "6+ (Large Family)"]
    family_stats = family_stats.reindex([cat for cat in category_order if cat in family_stats.index])
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with smaller bar width to prevent overlap
    colors = ['gold', 'lightcoral', 'lightblue', 'lightgreen']
    bars = plt.bar(range(len(family_stats)), family_stats['survival_rate'], 
                   color=colors[:len(family_stats)], 
                   alpha=0.8, edgecolor='darkblue', width=0.6)
    
    plt.title('Survival Rates by Family Size Group', fontsize=16, pad=20)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.xlabel('Family Size Group', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(range(len(family_stats)), family_stats.index, rotation=0, ha='center')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/family_survival.png')
    plt.close()

# Generate all visualizations
plot_gender_survival(df)
plot_class_survival(df)
plot_age_survival(df)
plot_family_survival(df)

print("Visualizations generated and saved in the 'images' folder.")