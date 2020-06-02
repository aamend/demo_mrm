# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Risk - modeling
# MAGIC 
# MAGIC **How to build models that move quickly through validation and audit**: *With regulators and policy makers seriously addressing the challenges of AI in finance and banks starting to demand more measurable profits around the use of data, data practices are forced to step up their game in the delivery of ML if they want to drive competitive insights reliable enough for business to trust and act upon. In this demo focused on credit risk analytics, we show how a unified data analytics platform brings a more disciplined and structured approach to commercial data science, reducing the model lifecycle process from 12 months to a few weeks.*
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_customer_etl">STAGE1</a>: Using Delta Lake for a curated and a 360 view of your customer and loan application data
# MAGIC + <a href="$./02_modeling">STAGE2</a>: Tracking experiments and registering models through MLflow capabilities, complying with ERMF and regulations
# MAGIC + <a href="$./03_template">STAGE3</a>: Template notebook that captures model risk related questions
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC In this notebook, we show how a simple data science problem (classification) could benefit from **Delta Lake** and **MLFlow** in order to bring reliability on your data and transparency in your insights. Specifically, we cover here some questions that any data practicitioner in financial services will have to comply from a model risk management perspective and how Databricks, as a unified data and analytics platform, can drastically help you reduce development-to-production time. 
# MAGIC 
# MAGIC We also want to show interaction with GitHub to run pre-commits hooks such as Black (code formatter) or flake8 (code style checker) to bring consistency in our deliverables. This assume we have created a file `.pre-commit-config.yaml` as follows
# MAGIC 
# MAGIC ```
# MAGIC repos:
# MAGIC -   repo: https://github.com/asottile/seed-isort-config
# MAGIC     rev: v1.9.3
# MAGIC     hooks:
# MAGIC     - id: seed-isort-config
# MAGIC -   repo: https://github.com/pre-commit/mirrors-isort
# MAGIC     rev: v4.3.21
# MAGIC     hooks:
# MAGIC     - id: isort
# MAGIC -   repo: https://github.com/ambv/black
# MAGIC     rev: stable
# MAGIC     hooks:
# MAGIC     - id: black
# MAGIC       language_version: python3.6
# MAGIC -   repo: https://github.com/pre-commit/pre-commit-hooks
# MAGIC     rev: v2.3.0
# MAGIC     hooks:
# MAGIC     - id: flake8
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Import statements
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pickle
import itertools
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from scipy import stats
import time

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1`: Data access
# MAGIC In this section, we access customer data with historical credit default **at the time of application**, leveraging our SCD type 2 pattern explained in previous notebook (<a href="$./01_customer_etl">see</a>). We also ensure our data remains consistent throughout our experiments using **Delta Lake** built-in functionalities (time travel).

# COMMAND ----------

# DBTITLE 1,Convert Spark to Pandas
delta_version = sql("DESCRIBE HISTORY antoine_fsi.credit_risk_gold").toPandas()['version'][0]

app_data = (
  spark \
    .read \
    .table('antoine_fsi.credit_risk_gold') \
    .filter("credit_type = 'CASH_LOANS'") \
    .drop('credit_type')
).toPandas()

app_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2`: Exploratory data analysis
# MAGIC In this section, we show how data scientists can collaboratively conduct standard exploratory data analysis using native python and third parties visualisations (such as Matplotlib). By capturing the work done during this step, alongside model development, it provides independent validation unit (part of your enterprise risk management framework) and audit with all the necessary context required to conduct model review and compliance with ERMF.

# COMMAND ----------

# DBTITLE 1,Age distribution against target variable
plt.figure(figsize = (12, 6))

# plot distribution for repaid loan with kernel density estimate
sns.kdeplot(app_data.loc[app_data['credit_default'] == 0, 'customer_age'], label = 'repaid', color='steelblue')
plt.hist(app_data.loc[app_data['credit_default'] == 0, 'customer_age'], bins=25, alpha=0.25, color='steelblue', density=True)

# plot distribution for default credit with kernel density estimate
sns.kdeplot(app_data.loc[app_data['credit_default'] == 1, 'customer_age'], label = 'default', color='coral')
plt.hist(app_data.loc[app_data['credit_default'] == 1, 'customer_age'], bins=25, alpha=0.25, color='coral', density=True)

plt.xlabel('Customer Age')
plt.ylabel('Density')
plt.title('Age distribution repayment')

# COMMAND ----------

# DBTITLE 1,Handle categorical data
# retrieve categorical variables
app_cat_features = app_data.select_dtypes(include='object')

# one-hot encoding of categorical variables
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(app_cat_features)
column_names = enc.get_feature_names(app_cat_features.columns)

# create dataset for categorical variable
app_cat_enc_features = pd.DataFrame(enc.fit_transform(app_cat_features).toarray(), columns=column_names)

# merge dataset with non categorical values
app_features = pd.merge(app_data.drop(app_cat_features.columns, axis=1), app_cat_enc_features, left_index=True, right_index=True)

# Feature names
features = app_features.columns

# Median imputation of missing values
imputer = SimpleImputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(app_features)

# Repeat with the scaler
scaler.fit(app_features)

app_features = imputer.transform(app_features)
app_features = scaler.transform(app_features)

# Transform both training and testing data
app_features = pd.DataFrame(app_features, columns=features)
app_features.head()

# COMMAND ----------

# DBTITLE 1,Correlations against target variable
# Find the correlations with the target
app_target_corr = pd.DataFrame(app_features.corr(method='pearson')['credit_default'].sort_values())
app_target_corr = app_target_corr.drop('credit_default')
app_target_corr['pearson'] = app_target_corr['credit_default']
app_target_corr['feature'] = app_target_corr.index

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 6))

# plot barchart of top 10 negative correlations against target variable (repaid credit)
sns.barplot(y= "pearson", x = "feature",data = app_target_corr.head(10), color='steelblue', ax=ax1, alpha=0.8)
ax1.tick_params(labelrotation=90)
ax1.set_title('Negative correlations')
ax1.set_ylabel('pearson')
ax1.set_xlabel('')

# plot barchart of top 10 positivie correlations against target variable (default credit)
sns.barplot(y= "pearson", x = "feature", data = app_target_corr.tail(10), color='coral', ax=ax2, alpha=0.8)
ax2.tick_params(labelrotation=90)
ax2.set_title('Positive correlations')
ax2.set_ylabel('pearson')
ax2.set_xlabel('')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP3`: Modeling
# MAGIC In this section, we evaluate multiple models and track each experiment on ml-flow, together with their accuracies (area under curve) and hyper parameters used. This shows model risk management and audit that multiple approaches were conducted and best model has been identified based on empirical results. 

# COMMAND ----------

# DBTITLE 1,Imbalanced dataset
# count how many record for target variable (default payment vs. repaid loan)
target_count = pd.DataFrame(app_data.groupby('credit_default')['credit_default'].count())
colors = ['steelblue', 'coral']
labels=['paid', 'default']

# plot pie chart for each group (target variable is 0 or 1)
fig, ax1 = plt.subplots(figsize = (12,8)) 
n = ax1.pie(target_count['credit_default'], colors=colors, startangle=90, autopct='%.1f%%', shadow = False) 
for _,ni in enumerate(n[0]):
    ni.set_alpha(0.5)

plt.title('Target repartition', fontsize = 15) 
ax1.legend(labels, loc = "upper right") 
plt.show()

# COMMAND ----------

# DBTITLE 1,Downsample negative class
# Separate majority and minority classes
df_negative = app_features[app_features.credit_default==0]
df_positive = app_features[app_features.credit_default==1]
 
# Upsample minority class
df_sampled = resample(df_negative, 
                                 replace=True,                  # sample with replacement
                                 n_samples=len(df_positive),    # to match majority class
                                 random_state=123)              # reproducible results
 
# Combine majority class with upsampled minority class
df_sampled = pd.concat([df_positive, df_sampled])

# COMMAND ----------

# DBTITLE 1,Split between train and test
y = df_sampled['credit_default']
X = df_sampled.drop('credit_default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print("Training set contains {} records".format(X_train.shape[0]))
print("Testing set contains {} records".format(X_test.shape[0]))

# COMMAND ----------

# DBTITLE 1,Evaluate multiple models through Grid search with 5-fold CV
clf_models = {}
clf_params = {}

clf_models['logistic_regression'] = LogisticRegression()
clf_params['logistic_regression'] = [{'C': [0.001, 0.05, 0.01]}]

clf_models['decision_tree'] = DecisionTreeClassifier()
clf_params['decision_tree'] = [{'max_depth': [5, 10, 50]}]

clf_models['random_forest'] = RandomForestClassifier(verbose = 0, n_jobs = -1)
clf_params['random_forest'] = [{'n_estimators': [50, 100, 200]}]

clf_models['gradient_boosting'] = GradientBoostingClassifier(max_features=2, max_depth=2, random_state=0)
clf_params['gradient_boosting'] = [{'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]}]

experiments = []

for _,name in enumerate(clf_models):
  
  # track each experiment on MLflow
  with mlflow.start_run(run_name=name):
    
    parameter_candidates = clf_params[name]
    algo = clf_models[name]
    
    # grid search with K-fold
    clf = GridSearchCV(
      estimator=algo, 
      param_grid=parameter_candidates, 
      n_jobs=-1, 
      scoring='roc_auc',
      cv=StratifiedKFold(5)
    )

    # Train models
    clf.fit(X_train, y_train)  
    
    # Log model 
    mlflow.sklearn.log_model(clf.best_estimator_, "model")
      
    # Retrieve best candidate
    model = clf.best_estimator_
    params = clf.best_params_
    score = clf.best_score_
    
    # Log each parameter used and AUC score
    mlflow.log_metric('roc_auc', score)
    for param in params.keys():
      mlflow.log_param(param, params[param])
      
    run_id = mlflow.active_run().info.run_id
    experiments.append([name, run_id, clf.best_params_, clf.best_score_])
    

experiments_df = pd.DataFrame(experiments, columns=['algo', 'run_id', 'params', 'auc'])
experiments_df.index = experiments_df['algo']
experiments_df = experiments_df.sort_values('auc', ascending=False)
experiments_df.drop('algo', axis=1)
experiments_df

# COMMAND ----------

# DBTITLE 1,Retrieve experiments and best model
# Retrieve all experiments provided in above list
models = [mlflow.sklearn.load_model(model_uri = "runs:/{}/model".format(run_id)) for run_id in experiments_df['run_id']]

# Retrieve best model based on AUC
best_algo = experiments_df['algo'][0]
best_run_id = experiments_df['run_id'][0]
best_auc = experiments_df['auc'][0]
best_run = mlflow.get_run(best_run_id)
best_model = models[0]

print("Best run is {}".format(best_run_id))
print("Best score is {}".format(best_auc))
print("Best model is {}".format(best_algo))

# COMMAND ----------

# DBTITLE 1,Receiver operating characteristic curve
plt.figure(figsize = (15,8))

# create our baseline model
ns_probs = [0 for _ in range(len(y_test))]
ns_fpr, ns_tpr, thresholds = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--')

# plot each model roc_curve
for model in models:
    pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pred)
    plt.plot(fpr, tpr, label = model.__class__.__name__)
    
plt.title('Receiver operating characteristic curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig("experiments.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP4`: Model delivery and compliance
# MAGIC In this section, we demonstrate how Databricks Unified Data Analytics Platform helps you attach all evidence and justification required for model validation and risk compliance. We use ml-registry to bring immutability and traceability in our model (single source of truth) that is linked to that particular notebook revision ID.

# COMMAND ----------

# DBTITLE 1,Feature importance
# Extract feature importances
feature_importance_values = best_model.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': feature_importance_values}).sort_values(by='importance', ascending=False)
feature_importances['importance_normalized'] = feature_importances['importance'] / feature_importances['importance'].sum()

# Plot top 20 features
plt.figure(figsize = (12,12))
ax = sns.barplot(x= "importance_normalized", y = "feature", data = feature_importances.head(20), palette=("Blues_r"), orient='h')

plt.xlabel('Normalized Importance')
plt.title('Feature Importances')
plt.ylabel('')
plt.savefig("feature_importance.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature description
df1 = app_target_corr[['pearson']].rename(columns = {'pearson': 'correlation'})
feature_importances.index = feature_importances['feature']
df2 = feature_importances

features_name = {
  'credit_amt': 'Amount borrowed',
  'credit_annuity': 'Credit annuity',
  'credit_annuity_pct': 'Percentage of salary that will cover credit annuity',
  'credit_term': 'Credit term in years',
  'credit_goods': 'Value of the good for which credit is asked',
  'customer_income': 'Customer income at the time of application',
  'customer_age': 'Customer age at the time of application',
  'days_employed': 'Number of days employed in current company',
  'customer_gender': 'Customer gender 1-M, 0-F',
  'own_car': 'Does customer own a car, 1:Y, 0:N',
  'own_car_age': 'How old is the car customer own',
  'own_realty': 'Does customer own a house or appartment, 1:Y, 0:N',
  'house_price': 'Price of the house customer lives in',
  'house_age': 'How old is the house customer own',
  'region_density': 'Density factor or area customer lives in, 0: urban, 1: rural',
  'children': 'Number of children customer has',
  'requests_bureau': 'How many requests done to bureau in last months, as proxy to number of applications',
  'bureau_1': 'Credit score for bureau agency, 0: bad, 1: good',
  'bureau_2': 'Credit score for bureau agency, 0: bad, 1: good',
  'bureau_3': 'Credit score for bureau agency, 0: bad, 1: good',
  'credit_type_CASH_LOANS': 'Cash credit',
  'credit_type_REVOLVING_LOANS': 'Revolving credit',
  'income_type_BUSINESSMAN': 'Income comes from business activities, 0: no, 1: yes',
  'income_type_COMMERCIAL_ASSOCIATE': 'Income comes from commercial activities, 0: no, 1: yes',
  'income_type_MATERNITY_LEAVE': 'Income comes from maternity leave, 0: no, 1: yes',
  'income_type_PENSIONER': 'Income comes from pension, 0: no, 1: yes',
  'income_type_STATE_SERVANT': 'Income comes from state, 0: no, 1: yes',
  'income_type_STUDENT': 'Income comes from student activities, 0: no, 1: yes',
  'income_type_WORKING': 'Income comes from salary, 0: no, 1: yes',
  'income_type_UNEMPLOYED': 'Income comes from unemployment, 0: no, 1: yes',
  'organization_ADVERTISING': 'Is customer working in advertising, 0: no, 1: yes',
  'organization_AGRICULTURE': 'Is customer working in agriculture, 0: no, 1: yes',
  'organization_BANK': 'Is customer working in banking, 0: no, 1: yes',
  'organization_BUSINESS': 'Is customer working in business, 0: no, 1: yes',
  'organization_CLEANING': 'Is customer working in cleaning, 0: no, 1: yes',
  'organization_CONSTRUCTION': 'Is customer working in construction, 0: no, 1: yes',
  'organization_CULTURE': 'Is customer working in culture, 0: no, 1: yes',
  'organization_ELECTRICITY': 'Is customer working in energy, 0: no, 1: yes',
  'organization_EMERGENCY': 'Is customer working in emergency, 0: no, 1: yes',
  'organization_GOVERNMENT': 'Is customer working in government, 0: no, 1: yes',
  'organization_HOTEL': 'Is customer working in hotel, 0: no, 1: yes',
  'organization_HOUSING': 'Is customer working in housing, 0: no, 1: yes',
  'organization_INDUSTRY': 'Is customer working in industry, 0: no, 1: yes',
  'organization_INSURANCE': 'Is customer working in insurance, 0: no, 1: yes',
  'organization_KINDERGARTEN': 'Is customer working in kindergarten, 0: no, 1: yes',
  'organization_LEGAL SERVICES': 'Is customer working in legal, 0: no, 1: yes',
  'organization_MEDICINE': 'Is customer working in advertising, 0: no, 1: yes',
  'organization_MILITARY': 'Is customer working in militart, 0: no, 1: yes',
  'organization_MOBILE': 'Is customer working in mobile, 0: no, 1: yes',
  'organization_OTHER': 'Is customer working in other, 0: no, 1: yes',
  'organization_POLICE': 'Is customer working in police, 0: no, 1: yes',
  'organization_POSTAL': 'Is customer working in postal, 0: no, 1: yes',
  'organization_REALTOR': 'Is customer working in realtor, 0: no, 1: yes',
  'organization_RELIGION': 'Is customer working in religion, 0: no, 1: yes',
  'organization_RESTAURANT': 'Is customer working in restaurant, 0: no, 1: yes',
  'organization_SCHOOL': 'Is customer working in school, 0: no, 1: yes',
  'organization_SECURITY': 'Is customer working in security, 0: no, 1: yes',
  'organization_SECURITY_MINISTRIES': 'Is customer working in ministries, 0: no, 1: yes',
  'organization_SELF_EMPLOYED': 'Is customer self employed, 0: no, 1: yes',
  'organization_SERVICES': 'Is customer working in services, 0: no, 1: yes',
  'organization_TELECOM': 'Is customer working in telecom, 0: no, 1: yes',
  'organization_TRADE': 'Is customer working in trade, 0: no, 1: yes',
  'organization_TRANSPORT': 'Is customer working in transport, 0: no, 1: yes',
  'organization_UNIVERSITY': 'Is customer working in university, 0: no, 1: yes',
  'organization_XNA': 'Is customer employment unknown, 0: no, 1: yes',
  'house_type_CO_OP_APARTMENT': 'Is customer cosharing their place, 0: no, 1: yes',
  'house_type_HOUSE_APARTMENT': 'Is customer living in their own place, 0: no, 1: yes',
  'house_type_MUNICIPAL_APARTMENT': 'Is customer living in council appartment, 0: no, 1: yes',
  'house_type_OFFICE_APARTMENT': 'Is customer living in office, 0: no, 1: yes',
  'house_type_RENTED_APARTMENT': 'Is customer renting appartment, 0: no, 1: yes',
  'house_type_WITH_PARENTS': 'Is customer living with their parents, 0: no, 1: yes',
  'education_ACADEMIC_DEGREE': 'Is customer highest education academic, 0: no, 1: yes',
  'education_HIGHER_EDUCATION': 'Is customer high education, 0: no, 1: yes',
  'education_INCOMPLETE_HIGHER': 'Is customer highest education imcomplete, 0: no, 1: yes',
  'education_LOWER_SECONDARY': 'Is customer highest education secondary, 0: no, 1: yes',
  'education_SECONDARY_SECONDARY_SPECIAL': 'Is customer highest education secondary imcomplete, 0: no, 1: yes',
  'family_status_CIVIL_MARRIAGE': 'Is customer married, 0: no, 1: yes',
  'family_status_MARRIED': 'Is customer married, 0: no, 1: yes',
  'family_status_SEPARATED': 'Is customer separated, 0: no, 1: yes',
  'family_status_SINGLE': 'Is customer single, 0: no, 1: yes',
  'family_status_UNKNOWN': 'Is customer family status unknown, 0: no, 1: yes',
  'family_status_WIDOW': 'Is customer a windower, 0: no, 1: yes',
}

df = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
df = pd.merge(df, pd.DataFrame.from_dict(features_name, orient='index', columns=['description']), left_index=True, right_index=True)
df = df[['feature', 'description', 'correlation', 'importance']]
df.to_csv('features.csv', index=False)

# COMMAND ----------

# DBTITLE 1,Log evidence to MLflow
# we do not wish to re-open run_id and therefore alter start / end time
# instead, we log artifact to existing run
client = mlflow.tracking.MlflowClient()

# demonstrate that our submitted model is the best fit
client.log_artifact(best_run_id, "experiments.png")

# store features
client.log_artifact(best_run_id, "features.csv")

# store features importance
client.log_artifact(best_run_id, "feature_importance.png")

# COMMAND ----------

# DBTITLE 1,Submit model to ml-registry
client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/model".format(best_run_id)
model_name = "credit_risk"
result = mlflow.register_model(
    model_uri,
    model_name
)

version = result.version

# model get registered asynchronously
# make sure model was registered
time.sleep(20)

description_md = """
# CREDIT RISK 

Predict default payment for unsecure lending application

![credit_default](/files/antoine.amend/images/credit_default.png)

## Business outcome

+ **We believe that** we could train a model to detect future default payment and automate loan application process
+ **Will result in** a faster application process and better understanding of the risk associated to unsecure lending
+ **We know we would be successful when** application time can be reduced by half

## Approach

+ We used customer demographics at the time of loan application
+ We evaluated 4 different models and found {} to be best fit
+ We evaluated feature importance

""".format(best_algo) 

# update model description
client.update_model_version(
    name=model_name,
    version=version,
    description=description_md
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP6`: Model risk management
# MAGIC 
# MAGIC As part of your enterprise risk management framework, model risk management ensures that your model complies with regulations and adheres to your organisations governance frameworks and AI policies. This usually ensures your model has been reviewed (independently) and that its financial and non-financial risks have been assessed and can take months. This long administrative process can be drastically reduced by leveraging native Databricks functionality to bring all the business and technical context to your IVU submission.
# MAGIC 
# MAGIC <img src="https://www.researchgate.net/profile/Andreas_Tsanakas/publication/277138848/figure/fig4/AS:668305383772163@1536347828089/The-Model-Risk-Management-Framework.png" width=500/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### &#x2705; What is the business purpose for which the model was built? 
# MAGIC *Model was trained to detect future default payment using historical data and customer demographics. It will be used to automate credit decisioning, reducing application process from weeks to minutes.*

# COMMAND ----------

# MAGIC %md
# MAGIC #### &#x2705; Is your model reproducible?
# MAGIC *We used a specific version number of our data stored on delta lake*

# COMMAND ----------

# MAGIC %md
# MAGIC #### &#x2705; Were alternative approaches evaluated and why is this approach better?
# MAGIC *We evaluated multiple models (as reported in this notebook) and identified Random Forest as best fit for that purpose*

# COMMAND ----------

# MAGIC %md
# MAGIC #### &#x2705; What features were used in that model? 
# MAGIC *We reported all features used in our model as per attached CSV*

# COMMAND ----------

# MAGIC %md
# MAGIC #### &#x2705; What is the importance of each feature to your model?
# MAGIC *We evaluated importance of each feature to our model and reported our findings in both png and csv attached*

# COMMAND ----------

# MAGIC %md
# MAGIC #### &#x2705; Has your model been versioned and stored centrally?
# MAGIC *We have stored model binary on ml-registry, linked to our original notebook and delta version*

# COMMAND ----------

# MAGIC %md
# MAGIC #### &#x274C; Is your model discriminatory to a specific population?
# MAGIC *It appears we have used gender and age as input features to our model*
# MAGIC 
# MAGIC In addition to being highly unethical, our **model is also not legal from a regulatory standpoint**

# COMMAND ----------

# DBTITLE 1,Gender discrimination
# predict our test dataset across male and female applicants
prediction_df = X_test.copy()[['customer_gender']]
prediction_df['default'] = best_model.predict(X_test)
prediction_df['customer_gender'] = np.where(prediction_df['customer_gender'] == 0, 'FEMALE', 'MALE')
prediction_df['default'] = np.where(prediction_df['default'] == 0, 'repaid', 'default')

# group negative / positive class for different gender
gender_matrix = pd.crosstab(prediction_df['customer_gender'], prediction_df['default'])
gender_matrix['total'] = gender_matrix['repaid'] + gender_matrix['default']
gender_matrix['default'] = gender_matrix['default'] / gender_matrix['total']
gender_matrix['repaid'] = gender_matrix['repaid'] / gender_matrix['total']

# plot both bar of negative / positive class for each gender
plt.style.use('seaborn-deep')
plt.figure(figsize = (6, 6))
plt.bar(gender_matrix.index, height = gender_matrix['repaid'], label='repaid', color='steelblue', alpha=0.8)
plt.bar(gender_matrix.index, height = gender_matrix['default'], bottom= gender_matrix['repaid'],label='default', color='coral', alpha=0.8)

plt.title('Gender bias')
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='upper right')
plt.show()

# COMMAND ----------

# DBTITLE 1,Would a same application would have been different for female applicants?
# retrieve all male applicants with propensity to default
propensity_default = X_test[best_model.predict(X_test) == 1]
propensity_default_male = propensity_default[propensity_default['customer_gender'] == 1.0]
initial_size = propensity_default_male.shape[0]

# would propensity be different if they were female applicants?
biased_df = propensity_default_male.copy()
biased_df['customer_gender'] = 0.0

biased_size = biased_df[best_model.predict(biased_df) == 1].shape[0]
displayHTML("<h2>{}% of rejected applications would have been accepted</h2>".format(int(100 * (initial_size - biased_size) / initial_size)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP7`: AI and ethics
# MAGIC 
# MAGIC AI ethics is a complicated topic that many of our customer are grappling with today. While we have no definitive Point of View on how best to tackle this issue, we believe that 1) being transparent about your approach and 2) being able to explain your data are 2 pre-requisite steps that **Delta Lake** and **MLflow** are uniquely positioned to help you with. Having the right open-source tools is the first step to addressing this challenge.
# MAGIC 
# MAGIC Addressing this challenge is not as simple as not including specific features such as gender. Failing to account for gender is precisely the problem. Research in algorithmic fairness has previously shown that considering gender actually helps mitigate gender bias. Ironically, though, doing so in the US is illegal. Now preliminary results from an ongoing study funded by the UN Foundation and the World Bank are once again challenging the fairness of gender-blind credit lending. Even when gender is not specified, **it can easily be deduced from other variables** that correlate highly with it. As a result, models trained on historical data stripped of gender still amplify past inequities. Because women were historically granted less credit, the algorithm learned to perpetuate that pattern. [source](https://www.technologyreview.com/2019/11/15/131935/theres-an-easy-way-to-make-lending-fairer-for-women-trouble-is-its-illegal/)

# COMMAND ----------

# DBTITLE 1,Example of discriminative proxy features
# extract organisation and gender from initial dataset
organization_df = app_data[['customer_gender', 'organization']]
organization_df['customer_gender'] = np.where(organization_df['customer_gender'] == 0, 'F', 'M')

# count organisations by gender and normalize
crosstab = pd.crosstab(organization_df['customer_gender'], organization_df['organization']).transpose()
crosstab['total'] = crosstab['F'] + crosstab['M']
crosstab['M'] = 100 * crosstab['M'] / crosstab['total']
crosstab['F'] = 100 * crosstab['F'] / crosstab['total']
crosstab = crosstab.sort_values(by='F', ascending=False)


plt.style.use('seaborn-deep')
plt.figure(figsize = (30, 6))

# plot organisations by gender representation (stacked to 100)
plt.bar(crosstab.index, height = crosstab['F'], label='F', alpha=0.8)
plt.bar(crosstab.index, height = crosstab['M'], bottom= crosstab['F'],label='M', alpha=0.8)

plt.title('Occupation')
plt.xlabel('')
plt.ylabel('')
plt.legend(loc="upper right")
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# DBTITLE 1,Gender-differentiated credit lending
# MAGIC %md
# MAGIC 
# MAGIC In a [2018 study](https://www.semanticscholar.org/paper/ADVANCES-IN-BIG-DATA-RESEARCH-IN-ECONOMICS-Fairness-Kleinberg-Ludwig/bbad34e7476371e3fc521aa657a7ef0faca2e77c), a collaboration between computer scientists and economists found that the best way to mitigate these issues was in fact to reintroduce characteristics like gender and race into the model. Doing so allows for more control to measure and reverse any manifested biases, resulting in more fairness overall. Preliminary findings suggested that **separating models for men and women would reduce gender bias further**. The study found that creating entirely separate creditworthiness models for men and women granted the majority of women more credit.

# COMMAND ----------

# DBTITLE 1,Evaluate a gender differentiated credit risk model
X_train_gender = X_train[X_train['customer_gender'] == 0]
y_train_gender = y_train[X_train['customer_gender'] == 0]
X_test_gender = X_test[X_test['customer_gender'] == 0]
y_test_gender = y_test[X_test['customer_gender'] == 0]

gender_model = RandomForestClassifier(verbose = 0, n_jobs = -1, n_estimators=200)
gender_model.fit(X_train_gender, y_train_gender)  

gender_model_pred = pd.DataFrame(gender_model.predict(X_test_gender), columns=['predictions'])
gender_model_pred['status'] = np.where(gender_model_pred['predictions'] == 0, 'repaid', 'default')
df1 = pd.DataFrame(gender_model_pred.groupby(['status']).size(), columns=['gender_model']).transpose()

model_pred = pd.DataFrame(best_model.predict(X_test_gender), columns=['predictions'])
model_pred['status'] = np.where(model_pred['predictions'] == 0.0, 'repaid', 'default')
df2 = pd.DataFrame(model_pred.groupby(['status']).size(), columns=['illegal_model']).transpose()

pd.concat([df1, df2])

# COMMAND ----------

# DBTITLE 0,Take away
# MAGIC %md
# MAGIC 
# MAGIC ## Take away
# MAGIC 
# MAGIC ** Customer trust is a key assets for a bank today**. Using customer data for data science is not an entitlement but a privilege. By unifying data and analytics with Databricks, we  showed how to address both the science and engineering challenges, enable collaboration and peer review, and bring transparency and reliability to both your model and data (through **ML-Flow** and **Delta Lake**). This ability to enable commercial, responsible and explainable use of data while adhering to existing risk frameworks, banks policies and regulations, is the key to driving digital transformation while safeguarding one of its most important assets: trust of customers.

# COMMAND ----------

