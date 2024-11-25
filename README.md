# Customer Churn Prediction

## Overview
Customer churn is a critical issue for businesses as retaining existing customers is often more cost-effective than acquiring new ones. This project uses machine learning techniques to predict customer churn, enabling businesses to identify at-risk customers and take proactive measures to retain them.

---

## Features
- **Data Preprocessing**: Handle missing values, perform feature scaling, and encode categorical variables.
- **Exploratory Data Analysis (EDA)**: Understand the data distribution and relationships between features using visualizations.
- **SMOTE**: Balance the dataset using Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance.
- **Machine Learning Models**: Train and evaluate multiple algorithms (e.g., Random Forest, Logistic Regression, and Gradient Boosting) for churn prediction.
- **Hyperparameter Tuning**: Optimize models using Grid Search or Randomized Search for better accuracy.
- **Model Evaluation**: Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization**: Create insightful visualizations like bar plots, heatmaps, and ROC curves to present results.

---

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Oversampling: Imbalanced-learn (SMOTE)

---

## Workflow
1. **Dataset**: Load and analyze the dataset containing customer features and churn labels.
2. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables using techniques like one-hot encoding.
   - Scale numerical features for uniformity.
3. **EDA**:
   - Visualize class distributions (e.g., Churn vs. Non-Churn).
   - Analyze feature correlations using heatmaps.
4. **SMOTE**:
   - Apply SMOTE to balance the dataset and handle class imbalance.
5. **Model Training**:
   - Train multiple models and compare their performances.
   - Use cross-validation for robust evaluation.
6. **Model Evaluation**:
   - Generate confusion matrices and classification reports.
   - Visualize performance using bar plots and accuracy comparisons.
7. **Deployment **:
   - Deploy the model using Streamlit for real-time predictions.

---

## Key Results
- Achieved **1.0 accuracy** and **1.0 F1-score** with the Random Forest model.
- Identified key features contributing to churn, such as tenure, monthly charges, and contract type.
