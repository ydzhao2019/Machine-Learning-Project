# Machine-Learning-Project
Machine Learning and Financial Application Project

Employed Machine Learning models such as Decision Tree, Random Forest, Logistic Regression, Naïve Bayes and SVM, as well as Ensemble models such as Bagging, Boosting and Stacking, to predict Market Movement at time t+1 using available financial data at time t, achieved precision 0.67 and recall 0.62

1. DataCleansing: get_data.py (most data is obtained from investing.com and Yahoo Finance)

2. FeatureEngineering: get_feature.py (create new features using raw data)

3. RawFeatureInput: select_feature.py (since we had too many features, we had to extract most important ones form trader's perspective)

4. ModelTraining: model_train_test.py (main part--model train and test)
