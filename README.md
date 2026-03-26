# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Data Read the dataset (food_items_binary.csv), select important features (Calories, Fat, Sugars, etc.), and define the target class.
2.Split the Dataset Divide the data into training (70%) and testing (30%) sets using train_test_split().
3.Scale the Features Apply StandardScaler to normalize the data so that all features have equal importance (important for SVM).
4.Train SVM with Hyperparameter Tuning Use SVC (Support Vector Classifier) and apply GridSearchCV to find the best parameters (C, kernel, gamma) using 5-fold cross-validation.
5.Evaluate the Model Predict test data, calculate accuracy, generate classification report, and display confusion matrix to check performance.

## Program:
```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat', 'Saturated Fat','Sugars', 'Dietary Fiber', 'Protein']
target='class'
X=data[features]
y=data[target]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm=SVC()
param_grid={
    'C':[0.1, 1, 10, 100],
    'kernel':['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search=GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model=grid_search.best_estimator_
print("Name: K Sadhana")
print("Register Number: 212225240128")
print("Best Parameters:",grid_search.best_params_)
y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Name: R Lenasri")
print("Register Number: 212225040199")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test, y_pred))
conf_matrix=confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="846" height="580" alt="image" src="https://github.com/user-attachments/assets/111211cd-47a3-432a-9f91-aba7842f148b" />
<img width="723" height="289" alt="image" src="https://github.com/user-attachments/assets/3fc95906-6214-4f4b-b482-f97f65a113a0" />
<img width="718" height="583" alt="image" src="https://github.com/user-attachments/assets/1464545c-b971-425f-a92c-841a6e417024" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
