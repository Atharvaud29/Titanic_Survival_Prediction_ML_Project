# Titanic_Survival_Prediction_ML_Project
The Titanic survival prediction project is a classic machine learning problem where the goal is to predict whether a passenger on the Titanic survived or not based on various features such as age, sex, ticket class, and more.
The dataset used in this project typically contains information about passengers who were aboard the Titanic, including whether they survived or not. The dataset is split into a training set and a test set. The training set is used to train the machine learning model, while the test set is used to evaluate the model's performance.

Goal ->
To build a machine learning model that can accurately predict whether a passenger survived or not based on the available features. The model can then be used to predict the survival of passengers in other similar situations.

Below is the explaination 

1. Import the libaraies -> Numpy
   Pandas
   Matplotlib
   sk-learn

2. Load the dataset -> This dataset include the parameter like PassengerId,	Survived,	Pclass,	Name,	Sex,	Age,	SibSp,	Parch,	Ticket,	Fare,	Cabin,	Embarked. This is the dataset of 891 rows × 12 columns.

3. Corrlation Heatmap ->

    ![image](https://github.com/user-attachments/assets/c416882e-1340-456d-b1d4-a725d6193e3e)

    ![image](https://github.com/user-attachments/assets/a8b7adb6-f929-4b74-9106-dd226a08c8b9)

   Now the size of the training dataset is : strat_train_set.info()
   we get the o/p as -> <class 'pandas.core.frame.DataFrame'>
   Int64Index: 712 entries, 624 to 341
   Data columns (total 12 columns):
   #   Column       Non-Null Count  Dtype  
   ---  ------       --------------  -----  
   0   PassengerId  712 non-null    int64  
   1   Survived     712 non-null    int64  
   2   Pclass       712 non-null    int64  
   3   Name         712 non-null    object 
   4   Sex          712 non-null    object 
   5   Age          569 non-null    float64
   6   SibSp        712 non-null    int64  
   7   Parch        712 non-null    int64  
   8   Ticket       712 non-null    object 
   9   Fare         712 non-null    float64
   10  Cabin        163 non-null    object 
   11  Embarked     710 non-null    object 
   dtypes: float64(2), int64(5), object(5)
   memory usage: 72.3+ KB

4. Estimators ->

       from sklearn.base import BaseEstimator, TransformerMixin
       from sklearn.impute import SimpleImputer

       class AgeImputer(BaseEstimator, TransformerMixin):

       def fit(self, X, y=None):
          return self

       def transform(self, X):
          imputer=SimpleImputer(strategy="mean")
          X['Age']=imputer.fit_transform(X[['Age']])
       return X

5. Encoding -> in this we are using the OneHotEncodeing
   [ from sklearn.preprocessing import OneHotEncoder ]
   

       class FeatureEncoder(BaseEstimator, TransformerMixin):

       def fit(self, X, y=None):
          return self

       def transform(self, X):
           encoder=OneHotEncoder()
           matrix=encoder.fit_transform(X[['Embarked']]).toarray()

        column_names=["C","S","Q","N"]

        for i in range(len(matrix.T)):
            X[column_names[i]]=matrix.T[i]

        matrix = encoder.fit_transform(X[['Sex']]).toarray()

        column_names =  ["Female", "Male"]

        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

         return X

6. Feature dropper -> we need to drop some parameters from the dataset such as ["Embarked", "Name", "Ticket", "Cabin", "Sex","N"]
  
7. Pipeline -> Pipeline allows you to sequentially apply a list of transformers to preprocess the data and, if desired, conclude the sequence with a final 
               predictor for predictive modeling.

   Intermediate steps of the pipeline must be transformers, that is, they must implement fit and transform methods. The final estimator only needs to 
   implement fit. The transformers in the pipeline can be cached using memory argument.

       from sklearn.pipeline import Pipeline

       pipeline = Pipeline([("ageimputer", AgeImputer()),
                           ("featureencoder", FeatureEncoder()),
                           ("featuredopper", FeatureDropper())])

     


       strat_train_set = pipeline.fit_transform(strat_train_set)

       strat_train_set

Then we will get the parameters in dataset like [ PassengerId,	Survived,	Pclass,	Age,	SibSp,	Parch,	Fare,	C,	S,	Q,	Female,	Male ]  then the dataset consist of 712 rows × 12 columns.

       strat_train_set.info()

o/p => 
       <class 'pandas.core.frame.DataFrame'>
       Int64Index: 712 entries, 624 to 341
       Data columns (total 12 columns):
       #   Column       Non-Null Count  Dtype  
       ---  ------       --------------  -----  
       0   PassengerId  712 non-null    int64  
       1   Survived     712 non-null    int64  
       2   Pclass       712 non-null    int64  
       3   Age          712 non-null    float64
       4   SibSp        712 non-null    int64  
       5   Parch        712 non-null    int64  
       6   Fare         712 non-null    float64
       7   C            712 non-null    float64
       8   S            712 non-null    float64
       9   Q            712 non-null    float64
       10  Female       712 non-null    float64
       11  Male         712 non-null    float64
       dtypes: float64(7), int64(5)
       memory usage: 72.3 KB

8. Model selection -> for this we will use RandomForestClassifier, GridSearchCV.

   GridSearchCV(cv=3, estimator=RandomForestClassifier(),
             param_grid=[{'max_depth': [None, 5, 10],
                          'min_samples_split': [2, 3, 4],
                          'n_estimators': [10, 100, 200, 500]}],
             return_train_score=True, scoring='accuracy')

10. Model Accuracy -> Accuracy of the model is 80%

11. Test data ->

        prod_final_clf = grid_search.best_estimator_

12. Import the test dataset -> the test dataset is given to our model. This will give the predict the Survived people through " PassengerId ".
    Now the dataset consist of the parameters like [ PassengerId,	Survived ] and the size is 418 rows × 2 columns.
