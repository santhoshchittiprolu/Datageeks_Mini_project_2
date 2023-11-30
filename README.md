# Datageeks_Mini_project_2
# XYZ Bank Term Deposit Marketing Analysis
 
## Introduction
 
This project involves analyzing XYZ Bank's term deposit marketing data using PySpark. The goal is to enhance marketing strategies by identifying potential customers more likely to subscribe to term deposits.
 
## Data Preparation
 
1. **SparkSession Initialization**: A SparkSession is created with the name "BankMarketingEDA".
 
2. **Data Loading and Cleaning**: Data is loaded from "XYZ_Bank_Deposit_Data_Classification.csv". Columns with special characters or spaces are renamed for consistency.
 
3. **Exploratory Data Analysis (EDA)**:
   - Numeric and categorical features are analyzed for their distributions and unique counts.
   - Univariate and bivariate analyses are performed to understand feature relationships.
   - No missing values are detected in the dataset.
 
4. **Feature Engineering**:
   - Age groups are created using UDFs to categorize customers.
   - StringIndexer and OneHotEncoder are used for categorical features.
   - VectorAssembler is applied to combine features into a single vector.
 
## Model Training and Evaluation
 
1. **Train/Test Split**: The dataset is split into 80% training and 20% testing sets.
 
2. **Models Applied**:
   - Logistic Regression
   - Random Forest
   - Gradient Boosted Trees (GBT)
   - Decision Tree
   - Support Vector Machine (SVM)
 
   Each model's performance is evaluated using AUC, accuracy, and confusion matrices.
 
3. **Clustering**:
   - K-Means clustering is applied to understand customer segments better.
 
## Conclusion
 
The analysis provides valuable insights into XYZ Bank's customer data and helps in strategizing targeted marketing campaigns. The models predict potential term deposit subscribers with varying degrees of accuracy, offering a range of options for different marketing approaches.