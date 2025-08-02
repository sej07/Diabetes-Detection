## Diabetes Detection using Support Vector Machine (SVM)

_This project predicts whether a patient is diabetic or not using the Pima Indians Diabetes Dataset. It demonstrates an end-to-end machine learning pipeline using the SVM classifier, covering data preprocessing, standardization, model training, evaluation, and building a simple predictive system._

#### Dataset Details
- Source: Kaggle (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Files Used: `diabetes.csv`
- Rows: 768
- **Columns:**
  - `Pregnancies` : Number of times pregnant
  - `Glucose` : Plasma glucose concentration
  - `BloodPressure`: Diastolic blood pressure (mm Hg)
  - `SkinThickness`: Triceps skin fold thickness (mm)
  -  `Insulin`: 2-Hour serum insulin (mu U/ml)
  - `BMI`: Body mass index (weight in kg/(height in m)^2)
  - `DiabetesPedigreeFunction`: Hereditary likelihood of diabetes
  - `Age`: Patient age
  - `Outcome`: Target variable: `1` = Diabetic, `0` = Not diabetic 


#### ML Workflow: 
1. Importing Libraries
    1. `numpy`, `pandas` for data handling  
    2. `sklearn` for preprocessing, SVM classifier, and evaluation  
2. Data Collection & Analysis
    1. Loaded the dataset using `pandas`
    2. Checked for missing/null values
    3. Replaced the missing values using their mean
3. Data Standardization
    1. Used `StandardScaler` to normalize the features   
4. Train-Test Split
    1. Used `train_test_split` to divide the dataset into training and testing sets (75:25 ratio)  
5.  Model Training
    1. Trained a Support Vector Classifier (SVC) from `sklearn.svm`  
    2. Used `linear` kernel
6. Evaluation
    1. Metrics used: `Accuracy` 
7. Predictive System
    1. Built a simple input-based system to classify a new patient record  
    2. Input: `[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]`  
    3. Output: `'Diabetic'` or `'Not Diabetic'`

#### Results
Train Accuracy: 0.7708
Test Accuracy: 0.7864

#### What I Learned
1. Applying SVM for binary classification
2. Importance of data standardization before using distance-based algorithms
3. How to interpret confusion matrix and classification metrics
4. Deploying a basic prediction system using user input
