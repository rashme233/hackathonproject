TCS HACKATHON 
1. Problem Statement: Predicting Credit Risk for Loan Applicants

Background: Financial institutions face significant challenges in assessing the creditworthiness of loan applicants. Accurate credit risk prediction is crucial for minimizing defaults and ensuring the stability of the lending system. The German Credit dataset provides a comprehensive set of features related to applicants' financial history, personal information, and loan details, making it an ideal resource for developing predictive models.
Objective: Develop a machine learning model to predict the credit risk of loan applicants using the German Credit dataset. The model should classify applicants into two categories: good credit risk and bad credit risk. Additionally, provide insights into the key factors influencing credit risk and suggests strategies for improving the credit evaluation process. Requirements:
Data Exploration and Preprocessing:
Analyze the dataset to understand the distribution of features and target variables.
Handle missing values, and outliers, and perform necessary data cleaning.
Engineer new features that could enhance model performance.
Model Development:
Select appropriate machine learning algorithms for classification.
Train and validate the model using suitable evaluation metrics (e.g., accuracy, precision, recall, F1-score).
Optimize the model through techniques such as hyperparameter tuning and cross-validation.
Model Interpretation and Insights:
Interpret the model's predictions and identify the most influential features.
Create visualizations to communicate findings effectively.
Provide actionable insights and recommendations for improving the credit evaluation process.
Presentation:
Prepare a comprehensive report detailing the methodology, results, and conclusions. Explain why the implemented approach was selected.
You may use Streamlit for UI. Submit the recording of the demo with voice-over of what has been achieved along with the code.

Solution :
The primary objective of this project was to analyze the German Credit Data dataset and build a predictive model for identifying the credit risk associated with customers. The goal was to classify customers as either ‘good’ or ‘bad’ risks based on various financial and demographic attributes. The project utilized multiple machine learning techniques, including decision trees, and compared their performance with more advanced classifiers like XGBoost.

This project was divided into two main parts, spread across two separate notebooks.

First Notebook: Data Preprocessing and Model Training (project1.ipynb)
The project began by importing essential libraries such as pandas and numpy for data manipulation, and matplotlib and seaborn for data visualization. These were followed by modules from sci-kit-learn for preprocessing, model training, and evaluation.

The dataset was loaded using pd.read_csv() and its structure was examined with head() and info() to understand the data types and detect any missing values. To handle these missing values, a SimpleImputer with the 'most_frequent' strategy was applied to the ‘Saving accounts’ and ‘Checking account’ columns, replacing missing values with the most common value in each column.

Next, categorical variables like ‘Sex’, ‘Housing’, ‘Saving accounts’, and ‘Purpose’ were encoded using LabelEncoder, transforming them into numerical formats that are compatible with machine learning models. A custom function, assign_credit_risk(), was created to assign a credit risk score to each customer. This score was based on several factors, such as credit amount, loan duration, account balances, job type, and age. Customers exceeding a predefined risk threshold were labeled as ‘bad’ risk, while others were classified as ‘good’. This classification was added as a new column in the dataset and encoded numerically.

To gain insights into the distribution of various features, Exploratory Data Analysis (EDA) was performed. This included visualizations of housing types, job categories, gender, and credit risk classifications, created using seaborn and matplotlib. To handle categorical features, OneHotEncoder was applied to the ‘Saving accounts’ and ‘Checking account’ columns, converting them into multiple binary columns and integrating them back into the dataset.

The features (X) and target variable (y) were separated, with ‘Credit Risk’ as the target. The dataset was split into training and testing subsets using an 80-20 ratio via train_test_split(). StandardScaler was applied to the feature set to standardize numerical values, ensuring better performance with models that are sensitive to scale.

Various machine learning models were trained and evaluated for credit risk prediction. These included Support Vector Machine (SVM), Decision Tree Classifier, Random Forest Classifier, K-Nearest Neighbors (KNN), and XGBoost. The performance of these models was evaluated using metrics such as precision, recall, F1-score, and accuracy. Additionally, confusion matrices were used to understand the distribution of true positives, false positives, true negatives, and false negatives. Based on these evaluations, it was concluded that the Decision Tree algorithm provided with the highest accuracy.

Second Notebook: Streamlit Web Application for Credit Risk Prediction
The second part of the project involved developing an interactive Streamlit web application to predict credit risk based on customer data. Streamlit is an open-source Python library that allows for the creation of real-time data applications with minimal effort.

The script starts by importing the necessary libraries, including Streamlit, pandas, numpy, and modules from sci-kit-learn. The German Credit Data dataset is then loaded using pd.read_csv() and cleaned by dropping any unnecessary columns, such as ‘Unnamed: 0’ if present.

Missing values in the ‘Saving accounts’ and ‘Checking account’ columns are handled using the SimpleImputer strategy, ensuring consistency in the data. Following this, categorical columns like ‘Sex’, ‘Housing’, ‘Saving accounts’, and ‘Purpose’ are encoded into numeric values using LabelEncoder.

The custom function assign_credit_risk() is then defined to assign a risk score to each customer. The function considers factors like credit amount, loan duration, account balances, job type, and age to determine the risk. If the total score is above a predefined threshold, the customer is classified as a ‘bad’ risk, otherwise as ‘good’.

The ‘Credit Risk’ labels are encoded numerically using LabelEncoder. OneHotEncoding is applied to the ‘Saving accounts’ and ‘Checking account’ columns, converting each category into binary columns. These new columns are appended to the dataset, and the original categorical columns are dropped.

The features (X) are separated from the target (y), and the data is split into training and testing subsets using an 80-20 ratio. StandardScaler is applied to standardize the numerical data, ensuring optimal performance during model prediction.

A Decision Tree Classifier is trained on the data and used for making predictions based on user inputs through the Streamlit interface. The application features an easy-to-use UI, where users can input customer details such as age, job type, credit amount, loan duration, housing status, and account categories. When the ‘Predict Credit Risk’ button is clicked, the application processes the inputs, scales them, and feeds them into the trained model for prediction.

Finally, the predicted credit risk (either ‘good’ or ‘bad’) is displayed on the interface, informing the user of the customer's credit risk classification.
[Additionally the Committed App.py is also given in the side bar only for reference as this will be automatically created when streamlit.ipynb is executed]
[Reference Ouputs for this Code is provided as well]
