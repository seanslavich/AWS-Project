#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Model (Predicting Interview Acceptance)

# In[29]:


import re
import boto3
import sagemaker
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
s3_conn = boto3.client('s3')

import pyspark
from pyspark.sql import SparkSession

sess = sagemaker.Session()

region = boto3.Session().region_name

# S3 bucket where the original mnist data is downloaded and stored.
#downloaded_data_bucket = f"sagemaker-sample-files"
#downloaded_data_prefix = "datasets/image/MNIST"

# S3 bucket for saving code and model artifacts.
# Feel free to specify a different bucket and prefix
#bucket = sess.default_bucket()
#prefix = "sagemaker/DEMO-linear-mnist"

role = get_execution_role()


# Master dataframe imported from S3:

# In[30]:


# def max_partition(bucket: str = None,prefix: str =None):
#     response = s3_conn.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
#     #print(response['CommonPrefixes'])
#     replace_path=prefix+'date_col='
#     print(replace_path)
#     ts=[]
#     for item in response['CommonPrefixes']:
#         #print(item)
#         ts.append(item['Prefix'].replace(replace_path,'').replace('/',''))
#     #print(ts)
#     max_partition=max(ts)
#     print("max parition is", max_partition)
#     s3_path="s3://"+bucket+"/"+prefix+"date_col="+max_partition+"/*"
#     print(s3_path)
#     return s3_path


# In[38]:


# max_s3path = max_partition('hiring-curated-bucket', 'master-table/test_csv/')


# In[39]:


# master_df_new = spark.read.csv(max_s3path,  header = True, sep=',')
# master_df_new


# In[40]:


master_df_new = pd.read_csv(max_s3path)
master_df_new


# Feature Engineering:

# In[41]:


#master_path = 's3://ml-analytics-data-2022/training-new/master-full-dataset.csv'
master_df_new = pd.read_csv("s3://hiring-curated-bucket/master-table/test_csv/date_col=2022-07-28-05-06-21.423/*")
master_df_new


# In[42]:


train_path = 's3://ml-analytics-data-2022/training-new/output_1659045804/training-dataset.csv'
train_df = pd.read_csv(train_path)
train_df


# In[43]:


test_path = 's3://ml-analytics-data-2022/training-new/output_1659045962/testing-dataset.csv'
test_df = pd.read_csv(test_path)
test_df


# In[44]:


train_encode_path = 's3://ml-analytics-data-2022/training-new/output_1659050090/training-encoded.csv'
train_encode_df = pd.read_csv(train_encode_path)
train_encode_df


# In[45]:


test_encode_path = 's3://ml-analytics-data-2022/training-new/output_1659050187/testing_encoded.csv'
test_encode_df = pd.read_csv(test_encode_path)
test_encode_df


# In[46]:


# input
x_train = train_encode_df.drop(['Accepted_or_Not?'], axis=1)
x_test = test_encode_df.drop(['Accepted_or_Not?'], axis=1)

# output
y_train = train_encode_df.iloc[:, 1].values
y_test = test_encode_df.iloc[:, 1].values


# In[47]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.20, random_state = 0)


# In[48]:


x_train.shape


# In[49]:


(y_test)


# In[50]:



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[51]:


train_acc = classifier.score(x_train, y_train)
print("The Accuracy for the Training Set is {}".format(train_acc*100)+"%")


# In[52]:


y_pred = classifier.predict(x_test)
y_pred


# In[53]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
print ("Confusion Matrix : \n", cm)


# In[54]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))


# In[55]:


acc_prob = classifier.predict_proba(x_test)[:,1]
acc_prob


# In[23]:


a


# In[34]:


# import sagemaker
# input_data_path = 's3://ml-analytics-data-2022/training-new/output_1659045962/testing-dataset.csv'
# output_data_path = 's3://ml-analytics-data-2022/Notebook-instance/'
# transform_job = sagemaker.transformer.Transformer(
#     model_name = classifier,
#     instance_count = 1,
#     instance_type = 'ml.m4.xlarge',
#     strategy = 'SingleRecord',
#     assemble_with = 'Line',
#     output_path = output_data_path,
#     base_transform_job_name='inference-pipelines-batch',
#     sagemaker_session=sagemaker.Session(),
#     accept = 'application/csv')
# transform_job.transform(data = input_data_path, 
#                         content_type = 'application/csv', 
#                         split_type = 'Line')


# In[35]:


# from sklearn import svm
# from sklearn import datasets
# clf = svm.SVC()
# X, y= datasets.load_iris(return_X_y=True)
# clf.fit(X, y)


# import pickle
# s = pickle.dumps(clf)
# clf2 = pickle.loads(s)
# clf2.predict(X[0:1])


# In[36]:


# from joblib import dump, load
# dump(clf, 'Logistic_Regression.joblib') 


# In[37]:


# clf = load('Logistic_Regression.joblib')
# clf


# In[61]:


test_df.to_csv(r's3://ml-analytics-data-2022/prediction-model-output/static-output/pred-output.csv', index=False, header=True)

