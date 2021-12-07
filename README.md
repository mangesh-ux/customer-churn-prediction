
# Customer Churn Prediction

This end to end Data Science project predicts whether a customer will churn based on data of a subscription based video watching platform.


## How to run the app?

Create a python3 environment (<=3.8.x)

```bash
  python3 -m venv venv
```

Activate environment

```bash
  source venv/bin/activate
```
Keyword "source" is not required for windows users.

Install Requirements

```bash
  pip3 install -r requirements.txt
```

Run streamlit app

```bash
  streamlit run streamlit.py
```

Or a simple flask app


```bash
   python3 engine.py
```
## Solution Methodology

 - We start with exploring data, the project repository has a notebook named 'EDA & Data preprocessing'
 - There are considerable amount of missing values that we have to deal with. For 1.75% of data, the class label that is "churn" variable is missing
 ![visualising_missing_values](https://imgur.com/2naTaVx.png)
 - We use various imputation techniques to impute feature variables and KNN classification algorithm to impute missing class variable
 - We perform data visualizations to find interesting insights from data.
 - We perform label encoding of categorical variables and standardization for all numerical variable
 - After all this is done, we create a reusable pickle file having X_train, X_test,y_train and y_test
 - we also pickle standardization scalar object that is fitted on X_train to perform transformation on unseen data
 - For model building and evaluation, we load these data and run them with different algorithms.
 - The performance is found to be good with maximum accuracy of 91% with RandomForest and Gradient Boosting classifier
 ![roc_curve](https://imgur.com/ZMVHbyd.png)
 - We pickle the randomforest to make it reusable in an app
 - Finally, we create an app with streamlit and/or flask that takes data as input perform featurization and use our serialised model to make predictions and give output as whether the customer will churn or not


## Demo

Streamlit Output:

![streamlit_app](https://imgur.com/QGnGiMr.png)

flask Output:

![flask_engine](https://imgur.com/Gzzh0L5.png)

