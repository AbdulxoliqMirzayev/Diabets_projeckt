# Diabets_projeckt

Diabetes Prediction with Random Forest and SMOTE Balancing
This project is focused on predicting diabetes using a Random Forest classifier. The model is optimized through hyperparameter tuning using GridSearchCV, and SMOTE is applied to balance the dataset to handle class imbalances.

Project Overview
The project aims to:

Preprocess the dataset with standard scaling for numerical features.
Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
Use Random Forest to predict whether a person has diabetes or not.
Optimize the model using grid search with cross-validation.
Evaluate the model's accuracy using metrics such as accuracy score, precision, recall, F1-score, and a confusion matrix.
Dataset
The dataset used in this project is the Pima Indians Diabetes Database. It contains the following features:

Pregnancies: Number of times pregnant.
Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
BloodPressure: Diastolic blood pressure (mm Hg).
SkinThickness: Triceps skinfold thickness (mm).
Insulin: 2-hour serum insulin (mu U/ml).
BMI: Body mass index (weight in kg/(height in m)^2).
DiabetesPedigreeFunction: A function which scores likelihood of diabetes based on family history.
Age: Age in years.
Outcome: Whether the person has diabetes (1) or not (0).
Steps and Workflow
Data Preprocessing:

Missing values are checked (no missing values in this dataset).
The features are standardized using StandardScaler to normalize the data.
Handling Imbalance with SMOTE:

SMOTE is applied to address the imbalance between positive and negative outcomes (i.e., having diabetes or not).
This step increases the representation of the minority class (diabetes) to improve the model's predictive performance.
Model Training with Random Forest:

A RandomForestClassifier is trained on the preprocessed and balanced data.
Hyperparameters such as n_estimators, max_depth, and min_samples_split are optimized using GridSearchCV.
Model Evaluation:

The optimized model is evaluated using various metrics, including accuracy score, precision, recall, and F1-score.
A confusion matrix is visualized using Seaborn to better understand the model's performance.
Visualization:

The confusion matrix is displayed in a heatmap for a clear visualization of true positives, true negatives, false positives, and false negatives.
Code Execution
Requirements
To run the project, you need the following Python libraries:

bash
Copy code
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
Steps
Clone the repository or download the code.
Ensure the dataset is loaded properly from the provided URL in the code.
Run the Python script:
bash
Copy code
python diabetes_prediction.py
Output
The accuracy score of the optimized Random Forest model will be displayed.
A classification report including precision, recall, and F1-score will be printed.
A confusion matrix visualized as a heatmap will be generated.
Example Output:
bash
Copy code
Optimallashtirilgan Random Forest aniqlik darajasi: 75.32%

              precision    recall  f1-score   support
           0       0.85      0.75      0.80        99
           1       0.63      0.76      0.69        55
    accuracy                           0.75       154
   macro avg       0.74      0.76      0.74       154
weighted avg       0.77      0.75      0.76       154
Confusion Matrix
The confusion matrix is visualized using a heatmap to show the model's performance:

True Positives (TP): Diabetics correctly classified as diabetic.
True Negatives (TN): Non-diabetics correctly classified as non-diabetic.
False Positives (FP): Non-diabetics incorrectly classified as diabetic.
False Negatives (FN): Diabetics incorrectly classified as non-diabetic.
