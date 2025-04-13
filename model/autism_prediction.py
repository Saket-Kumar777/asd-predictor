

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

"""**Data** **Loading** **and** **Exploring**"""

df = pd.read_csv("/content/drive/MyDrive/Dataset/train.csv")

df.shape

df.head()

df.info()

#convert age column to int
df["age"] = df["age"].astype(int)

for col in df.columns:
  feature = ["ID", "age", "result"]
  if col not in feature:
    print(col, df[col].unique())
    print("-"*50)

df = df.rename(columns = {'austim' : 'autism'})

# dropping id and age_desc column
df = df.drop(["ID", "age_desc"], axis = 1)
df.shape

mapping = {
    "Viet Nam": "Vietnam",
    "Americansamoa": "United States",
    "hong Kong": "China"
}

#replacing value in country column
df['contry_of_res'] = df['contry_of_res'].replace(mapping)

#target class distribution
df['Class/ASD'].value_counts()

"""**Insights:**
1. missing values in ethnicity & relation
2. age_desc column has only 1 unique value. so it is removed
3. fixed country names
4. identified class imbalnce in target col

**EDA**
"""

df.shape

df.describe()

"""**Univariate Analysis**  
<br>**Numerical Columns:**  
- age  
- result
"""

sns.set_theme(style = "darkgrid")

#histogram for age

sns.histplot(df['age'], kde = True)
plt.title("Age Distribution")
plt.show()

"""Box plot for indetifying outliers in the numerical columns"""

sns.boxplot(x = df['age'])
plt.title("Boxplot for Age")
plt.xlabel("Age")
plt.show()

sns.boxplot(x = df['result'])
plt.title("Boxplot for result")
plt.xlabel("result")
plt.show()

df.columns

categorical_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',  'gender',
       'ethnicity', 'jaundice', 'autism', 'contry_of_res', 'used_app_before',
        'relation']

for col in categorical_cols:
    sns.countplot(x = df[col])
    plt.title(f"Count plot for {col}")
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

df['ethnicity'] = df['ethnicity'].replace({'?':'Others', 'others': 'Others'})

df['relation'].unique()

df['relation'] = df['relation'].replace(
    {
        '?':'Others',
        'Relative':'Others',
        'Parent':'Others',
        'Health care professional':'Others'
    }
)

obj_columns = df.select_dtypes(include = 'object').columns

print(obj_columns)

#intialize a dictionary to store the encoders
encoders = {}

#apply label encoding and store the encoders
for col in obj_columns:
  label_encoder = LabelEncoder()
  df[col] = label_encoder.fit_transform(df[col])
  encoders[col] = label_encoder    # saving encoder col wise

  with open('enocoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

df.head()

"""**Bivariate analysis**"""

plt.figure(figsize = (15,15))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Correlation heatmap')
plt.show()

"""**Insights from EDA**

- there are few outliers in the numerical columns(age, result)
- there is a class imbalance in the target column
- there is a class imbalance in the categorical feature
- we dont have any highly correlated column
- performed label encoding and saved encoders

**Data Preprocessing**
"""

#function to replace the outliers with median
def replace_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median = df[column].median()

    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median, df[column])

    return df

# replace outliers in the age column
df = replace_outliers(df, 'age')

"""train test split"""

x = df.drop('Class/ASD', axis = 1)
y = df['Class/ASD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(y_train.shape)
print(y_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())

smote = SMOTE(random_state = 42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print(y_train_smote.value_counts())

"""**Model Training**

"""

#list of classifier
models = {
    "Decision Tree": DecisionTreeClassifier(random_state = 42),
    "Random Forest": RandomForestClassifier(random_state = 42),
    "XGBoost": XGBClassifier(random_state = 42)
}

#dictionary to store the cross validation result
cv_scores = {}

#perform 5_fold cv for each model

for model_name, model in models.items():
  print(f"Training {model_name}...")
  cv_score = cross_val_score(model, x_train_smote, y_train_smote, cv = 5, scoring = "accuracy")
  cv_scores[model_name] = cv_score
  print(f"{model_name} cross validation accuracy: {np.mean(cv_score):.2f}")
  print("-"*50)

cv_scores

"""**Model selection and hyperparameter tuning**"""

#Intialize models
decision_tree = DecisionTreeClassifier(random_state = 42)
random_forest = RandomForestClassifier(random_state = 42)
xgboost_classifier = XGBClassifier(random_state = 42)

#Hyperparameter grids for RandomizedSearchCV
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20 , 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap':[True, False]

}

param_grid_xgb = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

#hyperparameter tuning for 3 tree based models

#below steps can be automated by using a for loop or by using  a pipeline

#perform RandomizedSearchCV for each model
random_search_dt = RandomizedSearchCV(estimator = decision_tree, param_distributions = param_grid_decision_tree, n_iter = 20, cv = 5, scoring = 'f1', random_state = 42)
random_search_rf = RandomizedSearchCV(estimator = random_forest, param_distributions = param_grid_rf, n_iter = 20, cv = 5, scoring = 'f1', random_state = 42)
random_search_xgb = RandomizedSearchCV(estimator = xgboost_classifier, param_distributions = param_grid_xgb, n_iter = 20, cv = 5, scoring = 'f1', random_state = 42)

#fit the models
random_search_dt.fit(x_train_smote, y_train_smote)
random_search_rf.fit(x_train_smote, y_train_smote)
random_search_xgb.fit(x_train_smote, y_train_smote)

best_model = None
best_score = 0

if random_search_dt.best_score_ > best_score:
  best_score = random_search_dt.best_score_
  best_model = random_search_dt.best_estimator_

if random_search_rf.best_score_ > best_score:
  best_score = random_search_rf.best_score_
  best_model = random_search_rf.best_estimator_

if random_search_xgb.best_score_ > best_score:
  best_score = random_search_xgb.best_score_
  best_model = random_search_xgb.best_estimator_

print(f"Best Model: {best_model}")
print(f"Best Score: {best_score: .2f}")

#save the best model

with open('best_model.pkl', 'wb') as f:
  pickle.dump(best_model, f)

"""**Evaluation**"""

#evaluate on test data
y_test_pred = best_model.predict(x_test)
print(f"Accuracy on test data:\n {accuracy_score(y_test, y_test_pred)}")
print(f"Confusion matrix on test data:\n {confusion_matrix(y_test, y_test_pred)}")
print(f"Classification report on test data:\n {classification_report(y_test, y_test_pred)}")



