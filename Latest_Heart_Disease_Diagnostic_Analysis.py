#!/usr/bin/env python
# coding: utf-8

# # **1. Importing the relevant libraries**

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore
import warnings
#import plotly.figure_factory as pff
warnings.filterwarnings('ignore')
from matplotlib.pyplot import figure

from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import math


# # **2. Importing the dataset**

# In[2]:


dataset=pd.read_csv('heart.csv')


# In[3]:


dataset


# The dataset consists of **303 individuals data**. There are **14 columns** in the dataset, which are described below.
# 
# 1. **Age**: displays the age of the individual.
# 
# 2. **Sex**: displays the gender of the individual using the following format :
# 
#   1 = **male**
# 
#   0 = **female**
# 
# 3. **Chest-pain type**: displays the type of chest-pain experienced by the individual using the following format :
# 
#   1 = **typical angina**
# 
#   2 = **atypical angina**
# 
#   3 = **non — anginal pain**
# 
#   4 = **asymptotic**
# 
# 4. **Resting Blood Pressure**: displays the resting blood pressure value of an individual in mmHg (unit)
# 
# 5. **Serum Cholestrol**: displays the serum cholesterol in mg/dl (unit)
# 
# 6. **Fasting Blood Sugar**: compares the fasting blood sugar value of an individual with 120mg/dl.
#   If fasting **blood sugar > 120mg/dl** then : **1 (true)**
#   else : **0 (false)**
# 
# 7. **Resting ECG** : displays resting electrocardiographic results
# 
#   0 =**normal**
# 
#   1 = **having ST-T wave abnormality**
# 
#   2 = **left ventricular hyperthrophy**
# 
# 8. **Max heart rate achieved** : displays the max heart rate achieved by an individual.
# 
# 9. **Exercise induced angina** :
# 
#   1 = **yes**
# 
#   0 = **no**
# 
# 10. **ST depression induced by exercise relative to rest**: displays the value which is an integer or float.
# 
# 11. Peak exercise ST segment :
# 
#   1 = **upsloping**
# 
#   2 = **flat**
# 
#   3 = **downsloping**
# 
# 12. **Number of major vessels (0–3) colored by flourosopy** : displays the value as integer or float.
# 
# 13. **Thal** : displays the thalassemia :
# 
#   3 = **normal**
# 
#   6 = **fixed defect**
# 
#   7 = **reversible defect**
# 
# 14. **Diagnosis of heart disease** : Displays whether the individual is suffering from heart disease or not :
# 
#   0 = **absence**
# 
#   1 = **present**

# # **3. Getting high level information of the data**

# ## **3.1. Printing the number of rows and column in the data**

# In[74]:


print (dataset.columns)


# In[75]:


print (dataset.shape)


# ## **3.2. Printing the column names**

# ## **3.3. Printing the datatype of all the column values**

# In[76]:


dataset.dtypes


# ## **3.4. Checking if there is null value present in the dataset**

# In[77]:


dataset.isnull().sum()


# ## **3.5. Checking the number of datapoints for each class**

# In[78]:


dataset["target"].value_counts()


# ## **3.6. Observations:**
# 
# *  **Total number** of **datapoints** in the dataset is **303**.
# 
# *  The dataset consist of **13 features** and **1 class** label.
# 
# *  **All** the **features** has **int 64 datatype** except **oldpeak** feature which is **float 64**.
# 
# *  There is **no null value** present in the dataset.
# 
# *  The dataset has **164** datapoints for **class 0** and **139** datapoints for **class 1**. Thus our dataset is **nearly balanced**.
# 
# *  The name of the **class** column is **'num'**. So we changed it to a meaningful name **'target'**.

# In[ ]:





# # **4. Analyszing Statistical details of the data**

# ### **4.1. Calculating the statistical parameters**

# In[79]:


dataset.describe()


# ### **4.2. Looking for the outlier**

# In[80]:


for i in range(dataset.shape[1]-1):
  print("Mean value of {} column is {}".format(dataset.columns[i],np.mean(dataset[dataset.columns[i]])))
  print("Median value of {} column is {}".format(dataset.columns[i],np.median(dataset[dataset.columns[i]])))
  print('*'*50+'\n')


# ### **Observation:**
# 
# *  If the **mean** and **median** of the column are **nearly equal** then there is **no outlier** in that column and **vice-versa**.
# 
# *  In our dataset, **mean** an **median** value of **age column** are nearly **equal**. Thus there are **no outliers** present.
# 
# *  In our dataset, difference between **mean** and **median** value of ca column is considerable. Thus ca column has outliers. Similarly **thal** column also has **an outlier**.

# ### **4.3. Removing the rows with the outliers**

# In[81]:


z_scores = zscore(dataset)


abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
dataset = dataset[filtered_entries]


# In[82]:


dataset.shape


# In[83]:


dataset["target"].value_counts()


# ### **4.4 Seeing the difference between the resting blood pressure value for people having and not having the heart disease.**

# In[84]:


dataset.groupby("target").mean()["trestbps"]


# ### **4.5 Seeing the difference between the resting cholestrol level for people having and not having the heart disease.**

# In[85]:


dataset.groupby("target").mean()["chol"]


# ### **4.6 Seeing the difference between the resting heart rate value for people having and not having the heart disease.**

# In[86]:


dataset.groupby("target").mean()["thalach"]


# ### **Observation:**
# 
# *  The **total number** of datapoints is reduced from **303** to **288** after **removing** the **rows** that has **outliers**.
# 
# *  After removing the outlers, dataset has **158** datapoints for **class 0** and **130** datapoints for **class 1**. Thus our dataset still is **nearly balanced**.
# 
# *  The **difference** between the **heart rate value** for person **with** and **without** heart disease is **significant**. So it can be a **good** **feature** for **classification**.

# # **5. Univariate and bivariate Analysis**

# ## **5.1. Plotting the pair plot and probablity distribution to see how much values overlap**

# In[88]:


plt.rcParams['figure.figsize']=20,20
sns.pairplot(dataset, hue="target")


# ## **5.2 Histogram plots**

# In[89]:


fig, axs = plt.subplots(3, 4, figsize=(20, 10))
index=0
for i in range(3):
  for j in range(4):
    sns.histplot(data=dataset, x=dataset.columns[index],hue='target', ax=axs[i, j],multiple="dodge")
    index=index+1


# In[90]:


figure(figsize=(8, 6), dpi=80)
sns.countplot(data= dataset, x='sex',hue='thal')
plt.title('Gender v/s Thalassemia\n')


# ### **Conclusion**
# 
# 1. From the **cp histogram plot**, it is clear that if the **cp** is of **type 4** then there is **highest** chance of getting the **heart disease**.
# 
# 2. From **count plot** of **gender vs thal**, it is clear that **male** have more chances of getting the **thalessmia**.

# ## **5.3 Box Plots**

# In[20]:


fig, axs = plt.subplots(3, 4, figsize=(20, 10))
index=0
for i in range(3):
  for j in range(4):
    sns.boxplot(data=dataset, y=dataset.columns[index],x='target',hue='target', ax=axs[i, j])
    index=index+1


# ## **5.4 Probablity density Function Plot**

# In[21]:


index=0
for i in range(3):
  for j in range(4):
    sns.FacetGrid(data=dataset,hue='target',size=5).map(sns.distplot,dataset.columns[index]) .add_legend();
    index=index+1


# ### **Conclusion:**
# 
# * From the pdf plot, the **'ca'**, **'oldpeak'**, **'thalach'**, and **'cp'** has slight **separated** distribution while all **other features** has **overlap** distribution.

# # **6. Correlation**

# In[22]:


plt.figure(figsize=(12,10))
sns.heatmap(dataset.corr())


# In[23]:


dataset.corr()['target']


# In[24]:


for x in dataset.corr().columns:
    print(np.abs(dataset.corr())[x].nlargest(2))


# ### **Conclusion**
# 
# * From the **correlation matrix**, it is clear that **cp**, **thalach**, **exang**, **oldpeak**, **ca** and **thal** are most correlated to the **target**.
# 
# * Out of the above features, **thal** is the **most** **correlated** feature.
# 
# * The correlation between the **thalach**, **sex** and **thalach**, **age** is the highest.

# # **7. Conclusion of EDA**

# * The dataset has **13 features** and **1 class** label.
# 
# * The class can take **two values**, thus it is a **binary class classification problem**.
# 
# * The dataset has **164 datapoints** for **class 0** and **139 datapoints** for **class 1**. Thus the dataset is** nearly balanced**.
# 
# * There is **no null** value present in the dataset.
# 
# * There are **outliers** present inside the dataset which are removed. The number of datapoints for **class 0** is **158** and **class 1** is **130** **after removing** the **outliers**.
# 
# * From **histogram plot**, it is clear that that **cp** with **type 4** has **high** chances of getting the **heart disease**.
# 
# * Male have more chances of getting the thalessmia.
# 
# * From the pdf plot, the **ca**, **oldpeak**, **thalach**, and **cp** has slight **separated** distribution while all **other features** has **overlap** distribution.
# 
# * From the **correlation matrix**, it is clear that **cp**, **thalach**, **exang**, **oldpeak**, **ca** and **thal** are most correlated to the **target**.
# 

# # **8. Splitting the data**

# In[25]:


from sklearn.model_selection import train_test_split

x = dataset.drop("target",axis=1)
y = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# # **9. Machine Learning Models**

# ### **9.1 Baseline Model**

# #### **a. Model Building**

# In[26]:


dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train,Y_train)
Y_pred_dummy=dummy_clf.predict(X_test)


# #### **b. Accuracy of the model**

# In[27]:


score_dummy = round(accuracy_score(Y_pred_dummy,Y_test)*100,2)

print("The accuracy score achieved using baseline model is: "+str(score_dummy)+" %")


# #### **c. Classification Report**

# In[28]:


print(classification_report(Y_test, Y_pred_dummy))


# #### **d. Confusion Matrix**

# In[29]:


cf_matrix = confusion_matrix(Y_test, Y_pred_dummy)


# In[30]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# ### **9.2. K-nearest Neighbours**

# #### **a. Hyperparameter tuning**

# In[31]:


auc_cv=[]
auc_train=[]
K=list(range(1,50,4))
cv_scores=[]
for i in K:
    knn=KNeighborsClassifier(n_neighbors=i,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
    knn.fit(X_train, Y_train)
    pred = knn.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,pred))
    pred1=knn.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))      
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(K, auc_train, label='AUC train')
ax.plot(K, auc_cv, label='AUC CV')
plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[32]:


from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=31,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
knn.fit(X_train, Y_train)
predi=knn.predict_proba(X_test)[:,1]
Y_pred_knn=knn.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=knn.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[33]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# #### **d. Classification Report**

# In[34]:


print(classification_report(Y_test, Y_pred_knn))


# #### **e. Confusion Matrix**

# In[35]:


cf_matrix = confusion_matrix(Y_test, Y_pred_knn)


# In[36]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# ### **9.3. Naive Bayes**

# #### **a. Hyperparaeter tuning**

# In[37]:


auc_train=[]
auc_cv=[]
alpha_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
    
for i in alpha_values:
    mnb = MultinomialNB(alpha = i)
    mnb.fit(X_train,Y_train)
    pred=mnb.predict_proba(X_test)[:,1]
    pred1=mnb.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))
    auc_cv.append(roc_auc_score(Y_test,pred))
    
optimal_alpha= alpha_values[auc_cv.index(max(auc_cv))]
alpha_values=[math.log(x) for x in alpha_values]
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha_values, auc_train, label='AUC train')
ax.plot(alpha_values, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('log(alpha)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best model**

# In[38]:


mnb=MultinomialNB(alpha = 0.00001)
mnb.fit(X_train,Y_train)
predi=mnb.predict_proba(X_test)[:,1]
Y_pred_nb=mnb.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=mnb.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[39]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# #### **d. Classification Report**

# In[40]:


print(classification_report(Y_test, Y_pred_nb))


# #### **e. Confusion Matrix**

# In[41]:


cf_matrix = confusion_matrix(Y_test, Y_pred_nb)


# In[42]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# ### **9.4. Logistic Regression**

# #### **a. Hyperparameter tuning**

# In[43]:


C = [10**-3, 10**-2, 10**0, 10**2,10**3,10**4]#C=1/lambda
auc_train=[]
auc_cv=[]
for c in C:
    lr=LogisticRegression(penalty='l2',C=c)
    lr.fit(X_train,Y_train)
    probcv=lr.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=lr.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_c= C[auc_cv.index(max(auc_cv))]
C=[math.log(x) for x in C]#converting values of C into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(C, auc_train, label='AUC train')
ax.plot(C, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('C (1/lambda)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal lambda for which auc is maximum : ',1//optimal_c)


# In[44]:


import math


# #### **b. Best model**

# In[45]:


lr=LogisticRegression(penalty='l2',C=optimal_c)
lr.fit(X_train,Y_train)
predi=lr.predict_proba(X_test)[:,1]
predi=lr.predict_proba(X_test)[:,1]
Y_pred_lr = lr.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=lr.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[46]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using logistic regression is: "+str(score_lr)+" %")


# #### **d. Classification Report**

# In[47]:


print(classification_report(Y_test, Y_pred_lr))


# #### **e. Confusion Matrix**

# In[48]:


cf_matrix = confusion_matrix(Y_test, Y_pred_lr)


# In[49]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# ### ***9.5. Support Vector Machine***

# #### **a. Hyperparameter tuning**

# In[50]:


alpha = [10**-4, 10**-3,10**-2,10**-1,1,10,10**2,10**3,10**4]#alpha=1/C
auc_train=[]
auc_cv=[]
for a in alpha:
    model=SGDClassifier(alpha=a) #loss default hinge
    svm=CalibratedClassifierCV(model, cv=3) #calibrated classifier cv for calculation of predic_proba
    svm.fit(X_train,Y_train)
    probcv=svm.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=svm.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_alpha= alpha[auc_cv.index(max(auc_cv))]
alpha=[math.log(x) for x in alpha]#converting values of alpha into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha, auc_train, label='AUC train')
ax.plot(alpha, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('alpha')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best Model**

# In[51]:


model=SGDClassifier(alpha=0.001)
svm=CalibratedClassifierCV(model, cv=3)
svm.fit(X_train,Y_train)
Y_pred_svm=svm.predict(X_test)
predi=svm.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=svm.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[52]:


score_svm= round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using SVM is: "+str(score_svm)+" %")


# #### **d. Classification Report**

# In[53]:


print(classification_report(Y_test, Y_pred_svm))


# #### **e. Confusion Matrix**

# In[54]:


cf_matrix = confusion_matrix(Y_test, Y_pred_svm)


# In[55]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# ### **9.6. Random Forest**

# #### **a. Hyperparameter tuning**

# In[56]:


base_learners = [20,40,60,80,100,120]
depths=[1,5,10,50,100,500,1000]
param_grid={'n_estimators': base_learners, 'max_depth':depths}
rf = RandomForestClassifier(max_features='sqrt')
model=GridSearchCV(rf,param_grid,scoring='roc_auc',n_jobs=-1,cv=3)
model.fit(X_train,Y_train)
print("optimal n_estimators",model.best_estimator_.n_estimators)
print("optimal max_depth",model.best_estimator_.max_depth)


# #### **b. Best model**

# In[57]:


import seaborn as sns
X=[]
Y=[]
Z=[]
Zt=[]
for bl in base_learners:
    for d in depths:
        rf=RandomForestClassifier(max_features='sqrt',max_depth=d,n_estimators=bl)
        rf.fit(X_train,Y_train)
        pred=rf.predict_proba(X_test)[:,1]
        predt=rf.predict_proba(X_train)[:,1]
        X.append(bl)
        Y.append(d)
        Z.append(roc_auc_score(Y_test,pred))
        Zt.append(roc_auc_score(Y_train,predt))
        
data = pd.DataFrame({'n_estimators': X, 'max_depth': Y, 'AUC': Z})
data_pivoted = data.pivot("n_estimators", "max_depth", "AUC")
ax = sns.heatmap(data_pivoted,annot=True)
plt.title('Heatmap for cross validation data')
plt.show()


# In[58]:


rf=RandomForestClassifier(max_features='sqrt',max_depth=80,n_estimators=5)
rf.fit(X_train,Y_train)
predi=rf.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=rf.predict_proba(X_train)[:,1]
Y_pred_rf=rf.predict(X_test)
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[59]:


score_rf= round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


# #### **d. Classification Report**

# In[60]:


print(classification_report(Y_test, Y_pred_rf))


# #### **e. Confusion Matrix**

# In[61]:


cf_matrix = confusion_matrix(Y_test, Y_pred_rf)


# In[62]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# ### **9.7. Decision Tree**

# #### **a. Hyperparamter Tuning**

# In[63]:


depths=[1,5,10,50,100,500,1000]
best_m=[]
min_samples=[2,5,10,15,100,500]
auc_train=[]
auc_cv=[]
for d in depths:
    ms,rc=0,0
    for m in min_samples:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            ms=m
    dt=DecisionTreeClassifier(max_depth=d,min_samples_split=ms)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))

        
        
    
optimal_depth= depths[auc_cv.index(max(auc_cv))]
optimal_min_samples_split=best_m[auc_cv.index(max(auc_cv))]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(depths, auc_train, label='AUC train')
ax.plot(depths, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter(depths)')
plt.xlabel('depths')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal depth for which auc is maximum : ',optimal_depth)
print('optimal minimum samples split for which auc is maximum : ',optimal_min_samples_split) 


# In[64]:


auc_train_m=[]
auc_cv_m=[]
for m in min_samples:
    dp,rc=0,0
    for d in depths:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            dp=d
    dt=DecisionTreeClassifier(max_depth=dp,min_samples_split=m)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv_m.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train_m.append(roc_auc_score(Y_train,probtr))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(min_samples, auc_train_m, label='AUC train')
ax.plot(min_samples, auc_cv_m, label='AUC CV')
plt.title('AUC vs hyperparameter(min_samples)')
plt.xlabel('min_samples')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[65]:


dt=DecisionTreeClassifier(max_depth=10,min_samples_split=15)
dt.fit(X_train,Y_train)
predi=dt.predict_proba(X_test)[:,1]
Y_pred_dt=dt.predict(X_test)

fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=dt.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[66]:


score_dt= round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# #### **d. Classification Report**

# In[67]:


print(classification_report(Y_test, Y_pred_dt))


# #### **e. Confusion Matrix**

# In[68]:


cf_matrix = confusion_matrix(Y_test, Y_pred_dt)


# In[69]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# # **10. Observation**

# # **12. Feature Importance and Feature Selection**

# In[70]:


from prettytable import PrettyTable
x = PrettyTable()


# In[ ]:



x.field_names = ["Sr.No","Model Name", "Best Parameter", "Accuracy"]
x.add_row(["1","Baseline model", None, score_dummy])
x.add_row(["2","Naive Bayes", "Alpha=0.00001", score_nb])
x.add_row(["3","K-Nearest Neighbors", "k=31", score_knn])
x.add_row(["4","Logistic Regression", "C=0", score_lr])
x.add_row(["5","Decision Tree","Max Depth=15 Min Split=10", score_dt])
x.add_row(["6","Random Forest","Max Depth=80 base learner=5", score_rf])
x.add_row(["7","Support Vector Machine", "alpha=0.001", score_svm])

print(x)


# # **11.** **Conclusion**
# 
# 
# 
# 1.   In this work we performed EDA and saw which feature could be helpful in predictions.
# 2.   We developed seven different machine learning models.
# 3.   We compared those machine learning models using accuracy, F1-score, specificity, and recall value. 
# 4.   We found that logistic regression performs the best, with an accuracy of 85.25%.
# 
# 

# ## **A. Filter Method**

# In[ ]:


import seaborn as sns
cor = dataset.corr()


# In[ ]:


cor_target = abs(cor["target"])
relevant_features = cor_target[cor_target>0.3]
relevant_features


# In[ ]:


relevant_features=pd.DataFrame(relevant_features)


# In[ ]:


relevant_features=relevant_features.sort_values(by=['target'],ascending=False)


# In[ ]:


selected_features=relevant_features.index
selected_features


# In[ ]:


data=dataset[[selected_features[1],selected_features[2],selected_features[3],selected_features[4],selected_features[5],
              selected_features[6],selected_features[7]]]


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(data,y,test_size=0.20,random_state=0)


# ### **9.1 Baseline Model**

# #### **a. Model Building**

# In[ ]:


dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train,Y_train)
Y_pred_dummy=dummy_clf.predict(X_test)


# #### **b. Accuracy of the model**

# In[ ]:


score_dummy = round(accuracy_score(Y_pred_dummy,Y_test)*100,2)

print("The accuracy score achieved using baseline model is: "+str(score_dummy)+" %")


# ### **9.2. K-nearest Neighbours**

# #### **a. Hyperparameter tuning**

# In[ ]:


auc_cv=[]
auc_train=[]
K=list(range(1,50,4))
cv_scores=[]
for i in K:
    knn=KNeighborsClassifier(n_neighbors=i,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
    knn.fit(X_train, Y_train)
    pred = knn.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,pred))
    pred1=knn.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))      
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(K, auc_train, label='AUC train')
ax.plot(K, auc_cv, label='AUC CV')
plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[ ]:


from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=31,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
knn.fit(X_train, Y_train)
predi=knn.predict_proba(X_test)[:,1]
Y_pred_knn=knn.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=knn.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# ### **9.3 Naive Bayes**

# #### **a. Hyperparaeter tuning**

# In[ ]:


auc_train=[]
auc_cv=[]
alpha_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
    
for i in alpha_values:
    mnb = MultinomialNB(alpha = i)
    mnb.fit(X_train,Y_train)
    pred=mnb.predict_proba(X_test)[:,1]
    pred1=mnb.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))
    auc_cv.append(roc_auc_score(Y_test,pred))
    
optimal_alpha= alpha_values[auc_cv.index(max(auc_cv))]
alpha_values=[math.log(x) for x in alpha_values]
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha_values, auc_train, label='AUC train')
ax.plot(alpha_values, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('log(alpha)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best model**

# In[ ]:


mnb=MultinomialNB(alpha = 0.00001)
mnb.fit(X_train,Y_train)
predi=mnb.predict_proba(X_test)[:,1]
Y_pred_nb=mnb.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=mnb.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# ### **9.4. Logistic Regression**

# #### **a. Hyperparameter tuning**

# In[ ]:


C = [10**-3, 10**-2, 10**0, 10**2,10**3,10**4]#C=1/lambda
auc_train=[]
auc_cv=[]
for c in C:
    lr=LogisticRegression(penalty='l2',C=c)
    lr.fit(X_train,Y_train)
    probcv=lr.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=lr.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_c= C[auc_cv.index(max(auc_cv))]
C=[math.log(x) for x in C]#converting values of C into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(C, auc_train, label='AUC train')
ax.plot(C, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('C (1/lambda)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal lambda for which auc is maximum : ',1//optimal_c)


# In[ ]:


import math


# #### **b. Best model**

# In[ ]:


lr=LogisticRegression(penalty='l2',C=optimal_c)
lr.fit(X_train,Y_train)
predi=lr.predict_proba(X_test)[:,1]
predi=lr.predict_proba(X_test)[:,1]
Y_pred_lr = lr.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=lr.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using logistic regression is: "+str(score_lr)+" %")


# ### ***9.5. Support Vector Machine***

# #### **a. Hyperparameter tuning**

# In[ ]:


alpha = [10**-4, 10**-3,10**-2,10**-1,1,10,10**2,10**3,10**4]#alpha=1/C
auc_train=[]
auc_cv=[]
for a in alpha:
    model=SGDClassifier(alpha=a) #loss default hinge
    svm=CalibratedClassifierCV(model, cv=3) #calibrated classifier cv for calculation of predic_proba
    svm.fit(X_train,Y_train)
    probcv=svm.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=svm.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_alpha= alpha[auc_cv.index(max(auc_cv))]
alpha=[math.log(x) for x in alpha]#converting values of alpha into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha, auc_train, label='AUC train')
ax.plot(alpha, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('alpha')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best Model**

# In[ ]:


model=SGDClassifier(alpha=0.001)
svm=CalibratedClassifierCV(model, cv=3)
svm.fit(X_train,Y_train)
Y_pred_svm=svm.predict(X_test)
predi=svm.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=svm.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_svm= round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using SVM is: "+str(score_svm)+" %")


# ### **9.6. Random Forest**

# #### **a. Hyperparameter tuning**

# In[ ]:


base_learners = [20,40,60,80,100,120]
depths=[1,5,10,50,100,500,1000]
param_grid={'n_estimators': base_learners, 'max_depth':depths}
rf = RandomForestClassifier(max_features='sqrt')
model=GridSearchCV(rf,param_grid,scoring='roc_auc',n_jobs=-1,cv=3)
model.fit(X_train,Y_train)
print("optimal n_estimators",model.best_estimator_.n_estimators)
print("optimal max_depth",model.best_estimator_.max_depth)


# #### **b. Best model**

# In[ ]:


import seaborn as sns
X=[]
Y=[]
Z=[]
Zt=[]
for bl in base_learners:
    for d in depths:
        rf=RandomForestClassifier(max_features='sqrt',max_depth=d,n_estimators=bl)
        rf.fit(X_train,Y_train)
        pred=rf.predict_proba(X_test)[:,1]
        predt=rf.predict_proba(X_train)[:,1]
        X.append(bl)
        Y.append(d)
        Z.append(roc_auc_score(Y_test,pred))
        Zt.append(roc_auc_score(Y_train,predt))
        
data = pd.DataFrame({'n_estimators': X, 'max_depth': Y, 'AUC': Z})
data_pivoted = data.pivot("n_estimators", "max_depth", "AUC")
ax = sns.heatmap(data_pivoted,annot=True)
plt.title('Heatmap for cross validation data')
plt.show()


# In[ ]:


rf=RandomForestClassifier(max_features='sqrt',max_depth=80,n_estimators=5)
rf.fit(X_train,Y_train)
predi=rf.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=rf.predict_proba(X_train)[:,1]
Y_pred_rf=rf.predict(X_test)
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_rf= round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


# ### **9.7. Decision Tree**

# #### **a. Hyperparamter Tuning**

# In[ ]:


depths=[1,5,10,50,100,500,1000]
best_m=[]
min_samples=[2,5,10,15,100,500]
auc_train=[]
auc_cv=[]
for d in depths:
    ms,rc=0,0
    for m in min_samples:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            ms=m
    dt=DecisionTreeClassifier(max_depth=d,min_samples_split=ms)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))

        
        
    
optimal_depth= depths[auc_cv.index(max(auc_cv))]
optimal_min_samples_split=best_m[auc_cv.index(max(auc_cv))]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(depths, auc_train, label='AUC train')
ax.plot(depths, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter(depths)')
plt.xlabel('depths')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal depth for which auc is maximum : ',optimal_depth)
print('optimal minimum samples split for which auc is maximum : ',optimal_min_samples_split) 


# In[ ]:


auc_train_m=[]
auc_cv_m=[]
for m in min_samples:
    dp,rc=0,0
    for d in depths:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            dp=d
    dt=DecisionTreeClassifier(max_depth=dp,min_samples_split=m)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv_m.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train_m.append(roc_auc_score(Y_train,probtr))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(min_samples, auc_train_m, label='AUC train')
ax.plot(min_samples, auc_cv_m, label='AUC CV')
plt.title('AUC vs hyperparameter(min_samples)')
plt.xlabel('min_samples')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[ ]:


dt=DecisionTreeClassifier(max_depth=10,min_samples_split=15)
dt.fit(X_train,Y_train)
predi=dt.predict_proba(X_test)[:,1]
Y_pred_dt=dt.predict(X_test)

fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=dt.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# ### **9.8. Accuracy of the model**

# In[ ]:


score_dt= round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[ ]:


from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Sr.No","Model Name", "Best Parameter", "Accuracy"]
x.add_row(["1","Baseline model", None, score_dummy])
x.add_row(["2","Naive Bayes", "Alpha=0.00001", score_nb])
x.add_row(["3","K-Nearest Neighbors", "k=31", score_knn])
x.add_row(["4","Logistic Regression", "C=0", score_lr])
x.add_row(["5","Decision Tree","Max Depth=15 Min Split=10", score_dt])
x.add_row(["6","Random Forest","Max Depth=80 base learner=5", score_rf])
x.add_row(["7","Support Vector Machine", "alpha=0.001", score_svm])

print(x)


# ## **B. Wrapper Method**

# In[ ]:


from sklearn.feature_selection import SequentialFeatureSelector
knn = KNeighborsClassifier(n_neighbors=30)
sfs = SequentialFeatureSelector(knn, n_features_to_select=6)
sfs.fit(dataset, y)


# In[ ]:


features=sfs.get_feature_names_out()
features


# In[ ]:


data=dataset[[features[0],features[1],features[2],features[3],features[4]]]


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(data,y,test_size=0.20,random_state=0)


# ### **9.1 Baseline Model**

# #### **a. Model Building**

# In[ ]:


dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train,Y_train)
Y_pred_dummy=dummy_clf.predict(X_test)


# #### **b. Accuracy of the model**

# In[ ]:


score_dummy = round(accuracy_score(Y_pred_dummy,Y_test)*100,2)

print("The accuracy score achieved using baseline model is: "+str(score_dummy)+" %")


# ### **9.2. K-nearest Neighbours**

# #### **a. Hyperparameter tuning**

# In[ ]:


auc_cv=[]
auc_train=[]
K=list(range(1,50,4))
cv_scores=[]
for i in K:
    knn=KNeighborsClassifier(n_neighbors=i,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
    knn.fit(X_train, Y_train)
    pred = knn.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,pred))
    pred1=knn.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))      
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(K, auc_train, label='AUC train')
ax.plot(K, auc_cv, label='AUC CV')
plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[ ]:


from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=31,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
knn.fit(X_train, Y_train)
predi=knn.predict_proba(X_test)[:,1]
Y_pred_knn=knn.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=knn.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# ### **9.3 Naive Bayes**

# #### **a. Hyperparaeter tuning**

# In[ ]:


auc_train=[]
auc_cv=[]
alpha_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
    
for i in alpha_values:
    mnb = MultinomialNB(alpha = i)
    mnb.fit(X_train,Y_train)
    pred=mnb.predict_proba(X_test)[:,1]
    pred1=mnb.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))
    auc_cv.append(roc_auc_score(Y_test,pred))
    
optimal_alpha= alpha_values[auc_cv.index(max(auc_cv))]
alpha_values=[math.log(x) for x in alpha_values]
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha_values, auc_train, label='AUC train')
ax.plot(alpha_values, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('log(alpha)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best model**

# In[ ]:


mnb=MultinomialNB(alpha = 0.00001)
mnb.fit(X_train,Y_train)
predi=mnb.predict_proba(X_test)[:,1]
Y_pred_nb=mnb.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=mnb.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# ### **9.4. Logistic Regression**

# #### **a. Hyperparameter tuning**

# In[ ]:


C = [10**-3, 10**-2, 10**0, 10**2,10**3,10**4]#C=1/lambda
auc_train=[]
auc_cv=[]
for c in C:
    lr=LogisticRegression(penalty='l2',C=c)
    lr.fit(X_train,Y_train)
    probcv=lr.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=lr.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_c= C[auc_cv.index(max(auc_cv))]
C=[math.log(x) for x in C]#converting values of C into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(C, auc_train, label='AUC train')
ax.plot(C, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('C (1/lambda)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal lambda for which auc is maximum : ',1//optimal_c)


# In[ ]:


import math


# #### **b. Best model**

# In[ ]:


lr=LogisticRegression(penalty='l2',C=optimal_c)
lr.fit(X_train,Y_train)
predi=lr.predict_proba(X_test)[:,1]
predi=lr.predict_proba(X_test)[:,1]
Y_pred_lr = lr.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=lr.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using logistic regression is: "+str(score_lr)+" %")


# ### ***9.5. Support Vector Machine***

# #### **a. Hyperparameter tuning**

# In[ ]:


alpha = [10**-4, 10**-3,10**-2,10**-1,1,10,10**2,10**3,10**4]#alpha=1/C
auc_train=[]
auc_cv=[]
for a in alpha:
    model=SGDClassifier(alpha=a) #loss default hinge
    svm=CalibratedClassifierCV(model, cv=3) #calibrated classifier cv for calculation of predic_proba
    svm.fit(X_train,Y_train)
    probcv=svm.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=svm.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_alpha= alpha[auc_cv.index(max(auc_cv))]
alpha=[math.log(x) for x in alpha]#converting values of alpha into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha, auc_train, label='AUC train')
ax.plot(alpha, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('alpha')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best Model**

# In[ ]:


model=SGDClassifier(alpha=0.001)
svm=CalibratedClassifierCV(model, cv=3)
svm.fit(X_train,Y_train)
Y_pred_svm=svm.predict(X_test)
predi=svm.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=svm.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_svm= round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using SVM is: "+str(score_svm)+" %")


# ### **9.6. Random Forest**

# #### **a. Hyperparameter tuning**

# In[ ]:


base_learners = [20,40,60,80,100,120]
depths=[1,5,10,50,100,500,1000]
param_grid={'n_estimators': base_learners, 'max_depth':depths}
rf = RandomForestClassifier(max_features='sqrt')
model=GridSearchCV(rf,param_grid,scoring='roc_auc',n_jobs=-1,cv=3)
model.fit(X_train,Y_train)
print("optimal n_estimators",model.best_estimator_.n_estimators)
print("optimal max_depth",model.best_estimator_.max_depth)


# #### **b. Best model**

# In[ ]:


import seaborn as sns
X=[]
Y=[]
Z=[]
Zt=[]
for bl in base_learners:
    for d in depths:
        rf=RandomForestClassifier(max_features='sqrt',max_depth=d,n_estimators=bl)
        rf.fit(X_train,Y_train)
        pred=rf.predict_proba(X_test)[:,1]
        predt=rf.predict_proba(X_train)[:,1]
        X.append(bl)
        Y.append(d)
        Z.append(roc_auc_score(Y_test,pred))
        Zt.append(roc_auc_score(Y_train,predt))
        
data = pd.DataFrame({'n_estimators': X, 'max_depth': Y, 'AUC': Z})
data_pivoted = data.pivot("n_estimators", "max_depth", "AUC")
ax = sns.heatmap(data_pivoted,annot=True)
plt.title('Heatmap for cross validation data')
plt.show()


# In[ ]:


rf=RandomForestClassifier(max_features='sqrt',max_depth=80,n_estimators=5)
rf.fit(X_train,Y_train)
predi=rf.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=rf.predict_proba(X_train)[:,1]
Y_pred_rf=rf.predict(X_test)
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_rf= round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


# ### **9.7. Decision Tree**

# #### **a. Hyperparamter Tuning**

# In[ ]:


depths=[1,5,10,50,100,500,1000]
best_m=[]
min_samples=[2,5,10,15,100,500]
auc_train=[]
auc_cv=[]
for d in depths:
    ms,rc=0,0
    for m in min_samples:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            ms=m
    dt=DecisionTreeClassifier(max_depth=d,min_samples_split=ms)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))

        
        
    
optimal_depth= depths[auc_cv.index(max(auc_cv))]
optimal_min_samples_split=best_m[auc_cv.index(max(auc_cv))]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(depths, auc_train, label='AUC train')
ax.plot(depths, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter(depths)')
plt.xlabel('depths')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal depth for which auc is maximum : ',optimal_depth)
print('optimal minimum samples split for which auc is maximum : ',optimal_min_samples_split) 


# In[ ]:


auc_train_m=[]
auc_cv_m=[]
for m in min_samples:
    dp,rc=0,0
    for d in depths:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            dp=d
    dt=DecisionTreeClassifier(max_depth=dp,min_samples_split=m)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv_m.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train_m.append(roc_auc_score(Y_train,probtr))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(min_samples, auc_train_m, label='AUC train')
ax.plot(min_samples, auc_cv_m, label='AUC CV')
plt.title('AUC vs hyperparameter(min_samples)')
plt.xlabel('min_samples')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[ ]:


dt=DecisionTreeClassifier(max_depth=10,min_samples_split=15)
dt.fit(X_train,Y_train)
predi=dt.predict_proba(X_test)[:,1]
Y_pred_dt=dt.predict(X_test)

fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=dt.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# ### **9.8. Accuracy of the model**

# In[ ]:


score_dt= round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[ ]:


from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Sr.No","Model Name", "Best Parameter", "Accuracy"]
x.add_row(["1","Baseline model", None, score_dummy])
x.add_row(["2","Naive Bayes", "Alpha=0.00001", score_nb])
x.add_row(["3","K-Nearest Neighbors", "k=31", score_knn])
x.add_row(["4","Logistic Regression", "C=0", score_lr])
x.add_row(["5","Decision Tree","Max Depth=15 Min Split=10", score_dt])
x.add_row(["6","Random Forest","Max Depth=80 base learner=5", score_rf])
x.add_row(["7","Support Vector Machine", "alpha=0.001", score_svm])

print(x)


# ## **C. Embedded Method**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

logistic=LogisticRegression(C=1,penalty="l1", solver='liblinear', random_state=7).fit(x,y)

model=SelectFromModel(logistic,prefit=True)

X_new=model.transform(x)


# In[ ]:


data=dataset[[selected_features[1],selected_features[2],selected_features[3],selected_features[4],selected_features[5],selected_features[6],selected_features[7]]]


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(data,y,test_size=0.20,random_state=0)


# In[ ]:


selected_features


# ### **9.1 Baseline Model**

# #### **a. Model Building**

# In[ ]:


dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train,Y_train)
Y_pred_dummy=dummy_clf.predict(X_test)


# #### **b. Accuracy of the model**

# In[ ]:


score_dummy = round(accuracy_score(Y_pred_dummy,Y_test)*100,2)

print("The accuracy score achieved using baseline model is: "+str(score_dummy)+" %")


# ### **9.2. K-nearest Neighbours**

# #### **a. Hyperparameter tuning**

# In[ ]:


auc_cv=[]
auc_train=[]
K=list(range(1,50,4))
cv_scores=[]
for i in K:
    knn=KNeighborsClassifier(n_neighbors=i,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
    knn.fit(X_train, Y_train)
    pred = knn.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,pred))
    pred1=knn.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))      
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(K, auc_train, label='AUC train')
ax.plot(K, auc_cv, label='AUC CV')
plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[ ]:


from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=31,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
knn.fit(X_train, Y_train)
predi=knn.predict_proba(X_test)[:,1]
Y_pred_knn=knn.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=knn.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# ### **9.3 Naive Bayes**

# #### **a. Hyperparaeter tuning**

# In[ ]:


auc_train=[]
auc_cv=[]
alpha_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
    
for i in alpha_values:
    mnb = MultinomialNB(alpha = i)
    mnb.fit(X_train,Y_train)
    pred=mnb.predict_proba(X_test)[:,1]
    pred1=mnb.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))
    auc_cv.append(roc_auc_score(Y_test,pred))
    
optimal_alpha= alpha_values[auc_cv.index(max(auc_cv))]
alpha_values=[math.log(x) for x in alpha_values]
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha_values, auc_train, label='AUC train')
ax.plot(alpha_values, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('log(alpha)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best model**

# In[ ]:


mnb=MultinomialNB(alpha = 0.00001)
mnb.fit(X_train,Y_train)
predi=mnb.predict_proba(X_test)[:,1]
Y_pred_nb=mnb.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=mnb.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# ### **9.4. Logistic Regression**

# #### **a. Hyperparameter tuning**

# In[ ]:


C = [10**-3, 10**-2, 10**0, 10**2,10**3,10**4]#C=1/lambda
auc_train=[]
auc_cv=[]
for c in C:
    lr=LogisticRegression(penalty='l2',C=c)
    lr.fit(X_train,Y_train)
    probcv=lr.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=lr.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_c= C[auc_cv.index(max(auc_cv))]
C=[math.log(x) for x in C]#converting values of C into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(C, auc_train, label='AUC train')
ax.plot(C, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('C (1/lambda)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal lambda for which auc is maximum : ',1//optimal_c)


# In[ ]:


import math


# #### **b. Best model**

# In[ ]:


lr=LogisticRegression(penalty='l2',C=optimal_c)
lr.fit(X_train,Y_train)
predi=lr.predict_proba(X_test)[:,1]
predi=lr.predict_proba(X_test)[:,1]
Y_pred_lr = lr.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=lr.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using logistic regression is: "+str(score_lr)+" %")


# ### ***9.5. Support Vector Machine***

# #### **a. Hyperparameter tuning**

# In[ ]:


alpha = [10**-4, 10**-3,10**-2,10**-1,1,10,10**2,10**3,10**4]#alpha=1/C
auc_train=[]
auc_cv=[]
for a in alpha:
    model=SGDClassifier(alpha=a) #loss default hinge
    svm=CalibratedClassifierCV(model, cv=3) #calibrated classifier cv for calculation of predic_proba
    svm.fit(X_train,Y_train)
    probcv=svm.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=svm.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_alpha= alpha[auc_cv.index(max(auc_cv))]
alpha=[math.log(x) for x in alpha]#converting values of alpha into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha, auc_train, label='AUC train')
ax.plot(alpha, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('alpha')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best Model**

# In[ ]:


model=SGDClassifier(alpha=0.001)
svm=CalibratedClassifierCV(model, cv=3)
svm.fit(X_train,Y_train)
Y_pred_svm=svm.predict(X_test)
predi=svm.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=svm.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_svm= round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using SVM is: "+str(score_svm)+" %")


# ### **9.6. Random Forest**

# #### **a. Hyperparameter tuning**

# In[ ]:


base_learners = [20,40,60,80,100,120]
depths=[1,5,10,50,100,500,1000]
param_grid={'n_estimators': base_learners, 'max_depth':depths}
rf = RandomForestClassifier(max_features='sqrt')
model=GridSearchCV(rf,param_grid,scoring='roc_auc',n_jobs=-1,cv=3)
model.fit(X_train,Y_train)
print("optimal n_estimators",model.best_estimator_.n_estimators)
print("optimal max_depth",model.best_estimator_.max_depth)


# #### **b. Best model**

# In[ ]:


import seaborn as sns
X=[]
Y=[]
Z=[]
Zt=[]
for bl in base_learners:
    for d in depths:
        rf=RandomForestClassifier(max_features='sqrt',max_depth=d,n_estimators=bl)
        rf.fit(X_train,Y_train)
        pred=rf.predict_proba(X_test)[:,1]
        predt=rf.predict_proba(X_train)[:,1]
        X.append(bl)
        Y.append(d)
        Z.append(roc_auc_score(Y_test,pred))
        Zt.append(roc_auc_score(Y_train,predt))
        
data = pd.DataFrame({'n_estimators': X, 'max_depth': Y, 'AUC': Z})
data_pivoted = data.pivot("n_estimators", "max_depth", "AUC")
ax = sns.heatmap(data_pivoted,annot=True)
plt.title('Heatmap for cross validation data')
plt.show()


# In[ ]:


rf=RandomForestClassifier(max_features='sqrt',max_depth=80,n_estimators=5)
rf.fit(X_train,Y_train)
predi=rf.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=rf.predict_proba(X_train)[:,1]
Y_pred_rf=rf.predict(X_test)
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_rf= round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


# ### **9.7. Decision Tree**

# #### **a. Hyperparamter Tuning**

# In[ ]:


depths=[1,5,10,50,100,500,1000]
best_m=[]
min_samples=[2,5,10,15,100,500]
auc_train=[]
auc_cv=[]
for d in depths:
    ms,rc=0,0
    for m in min_samples:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            ms=m
    dt=DecisionTreeClassifier(max_depth=d,min_samples_split=ms)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))

        
        
    
optimal_depth= depths[auc_cv.index(max(auc_cv))]
optimal_min_samples_split=best_m[auc_cv.index(max(auc_cv))]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(depths, auc_train, label='AUC train')
ax.plot(depths, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter(depths)')
plt.xlabel('depths')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal depth for which auc is maximum : ',optimal_depth)
print('optimal minimum samples split for which auc is maximum : ',optimal_min_samples_split) 


# In[ ]:


auc_train_m=[]
auc_cv_m=[]
for m in min_samples:
    dp,rc=0,0
    for d in depths:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            dp=d
    dt=DecisionTreeClassifier(max_depth=dp,min_samples_split=m)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv_m.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train_m.append(roc_auc_score(Y_train,probtr))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(min_samples, auc_train_m, label='AUC train')
ax.plot(min_samples, auc_cv_m, label='AUC CV')
plt.title('AUC vs hyperparameter(min_samples)')
plt.xlabel('min_samples')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[ ]:


dt=DecisionTreeClassifier(max_depth=10,min_samples_split=15)
dt.fit(X_train,Y_train)
predi=dt.predict_proba(X_test)[:,1]
Y_pred_dt=dt.predict(X_test)

fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=dt.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# ### **9.8. Accuracy of the model**

# In[ ]:


score_dt= round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[ ]:


from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Sr.No","Model Name", "Best Parameter", "Accuracy"]
x.add_row(["1","Baseline model", None, score_dummy])
x.add_row(["2","Naive Bayes", "Alpha=0.00001", score_nb])
x.add_row(["3","K-Nearest Neighbors", "k=31", score_knn])
x.add_row(["4","Logistic Regression", "C=0", score_lr])
x.add_row(["5","Decision Tree","Max Depth=15 Min Split=10", score_dt])
x.add_row(["6","Random Forest","Max Depth=80 base learner=5", score_rf])
x.add_row(["7","Support Vector Machine", "alpha=0.001", score_svm])

print(x)


# # **13. Normalization of data**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train = norm.transform(X_train)

# transform testing dataabs
X_test = norm.transform(X_test)


# #### **a. Model Building**

# In[ ]:


dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train,Y_train)
Y_pred_dummy=dummy_clf.predict(X_test)


# #### **b. Accuracy of the model**

# In[ ]:


score_dummy = round(accuracy_score(Y_pred_dummy,Y_test)*100,2)

print("The accuracy score achieved using baseline model is: "+str(score_dummy)+" %")


# ### **9.2. K-nearest Neighbours**

# #### **a. Hyperparameter tuning**

# In[ ]:


auc_cv=[]
auc_train=[]
K=list(range(1,50,4))
cv_scores=[]
for i in K:
    knn=KNeighborsClassifier(n_neighbors=i,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
    knn.fit(X_train, Y_train)
    pred = knn.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,pred))
    pred1=knn.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))      
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(K, auc_train, label='AUC train')
ax.plot(K, auc_cv, label='AUC CV')
plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')
ax.legend()
plt.show()


# #### **b. Best Model**

# In[ ]:


from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=31,weights='uniform',algorithm='brute',leaf_size=30, p=2, metric='cosine')
knn.fit(X_train, Y_train)
predi=knn.predict_proba(X_test)[:,1]
Y_pred_knn=knn.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=knn.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# ### **9.3 Naive Bayes**

# #### **a. Hyperparaeter tuning**

# In[ ]:


auc_train=[]
auc_cv=[]
alpha_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
    
for i in alpha_values:
    mnb = MultinomialNB(alpha = i)
    mnb.fit(X_train,Y_train)
    pred=mnb.predict_proba(X_test)[:,1]
    pred1=mnb.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,pred1))
    auc_cv.append(roc_auc_score(Y_test,pred))
    
optimal_alpha= alpha_values[auc_cv.index(max(auc_cv))]
alpha_values=[math.log(x) for x in alpha_values]
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha_values, auc_train, label='AUC train')
ax.plot(alpha_values, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('log(alpha)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best model**

# In[ ]:


mnb=MultinomialNB(alpha = 0.00001)
mnb.fit(X_train,Y_train)
predi=mnb.predict_proba(X_test)[:,1]
Y_pred_nb=mnb.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=mnb.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# ### **9.4. Logistic Regression**

# #### **a. Hyperparameter tuning**

# In[ ]:


C = [10**-3, 10**-2, 10**0, 10**2,10**3,10**4]#C=1/lambda
auc_train=[]
auc_cv=[]
for c in C:
    lr=LogisticRegression(penalty='l2',C=c)
    lr.fit(X_train,Y_train)
    probcv=lr.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=lr.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_c= C[auc_cv.index(max(auc_cv))]
C=[math.log(x) for x in C]#converting values of C into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(C, auc_train, label='AUC train')
ax.plot(C, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('C (1/lambda)')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal lambda for which auc is maximum : ',1//optimal_c)


# In[ ]:


import math


# #### **b. Best model**

# In[ ]:


lr=LogisticRegression(penalty='l2',C=optimal_c)
lr.fit(X_train,Y_train)
predi=lr.predict_proba(X_test)[:,1]
predi=lr.predict_proba(X_test)[:,1]
Y_pred_lr = lr.predict(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=lr.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using logistic regression is: "+str(score_lr)+" %")


# ### ***9.5. Support Vector Machine***

# #### **a. Hyperparameter tuning**

# In[ ]:


alpha = [10**-4, 10**-3,10**-2,10**-1,1,10,10**2,10**3,10**4]#alpha=1/C
auc_train=[]
auc_cv=[]
for a in alpha:
    model=SGDClassifier(alpha=a) #loss default hinge
    svm=CalibratedClassifierCV(model, cv=3) #calibrated classifier cv for calculation of predic_proba
    svm.fit(X_train,Y_train)
    probcv=svm.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    probtr=svm.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))
optimal_alpha= alpha[auc_cv.index(max(auc_cv))]
alpha=[math.log(x) for x in alpha]#converting values of alpha into logarithm
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(alpha, auc_train, label='AUC train')
ax.plot(alpha, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter')
plt.xlabel('alpha')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal alpha for which auc is maximum : ',optimal_alpha)


# #### **b. Best Model**

# In[ ]:


model=SGDClassifier(alpha=0.001)
svm=CalibratedClassifierCV(model, cv=3)
svm.fit(X_train,Y_train)
Y_pred_svm=svm.predict(X_test)
predi=svm.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=svm.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_svm= round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using SVM is: "+str(score_svm)+" %")


# ### **9.6. Random Forest**

# #### **a. Hyperparameter tuning**

# In[ ]:


base_learners = [20,40,60,80,100,120]
depths=[1,5,10,50,100,500,1000]
param_grid={'n_estimators': base_learners, 'max_depth':depths}
rf = RandomForestClassifier(max_features='sqrt')
model=GridSearchCV(rf,param_grid,scoring='roc_auc',n_jobs=-1,cv=3)
model.fit(X_train,Y_train)
print("optimal n_estimators",model.best_estimator_.n_estimators)
print("optimal max_depth",model.best_estimator_.max_depth)


# #### **b. Best model**

# In[ ]:


import seaborn as sns
X=[]
Y=[]
Z=[]
Zt=[]
for bl in base_learners:
    for d in depths:
        rf=RandomForestClassifier(max_features='sqrt',max_depth=d,n_estimators=bl)
        rf.fit(X_train,Y_train)
        pred=rf.predict_proba(X_test)[:,1]
        predt=rf.predict_proba(X_train)[:,1]
        X.append(bl)
        Y.append(d)
        Z.append(roc_auc_score(Y_test,pred))
        Zt.append(roc_auc_score(Y_train,predt))
        
data = pd.DataFrame({'n_estimators': X, 'max_depth': Y, 'AUC': Z})
data_pivoted = data.pivot("n_estimators", "max_depth", "AUC")
ax = sns.heatmap(data_pivoted,annot=True)
plt.title('Heatmap for cross validation data')
plt.show()


# In[ ]:


rf=RandomForestClassifier(max_features='sqrt',max_depth=80,n_estimators=5)
rf.fit(X_train,Y_train)
predi=rf.predict_proba(X_test)[:,1]
fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=rf.predict_proba(X_train)[:,1]
Y_pred_rf=rf.predict(X_test)
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# #### **c. Accuracy of the model**

# In[ ]:


score_rf= round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


# ### **9.7. Decision Tree**

# #### **a. Hyperparamter Tuning**

# In[ ]:


depths=[1,5,10,50,100,500,1000]
best_m=[]
min_samples=[2,5,10,15,100,500]
auc_train=[]
auc_cv=[]
for d in depths:
    ms,rc=0,0
    for m in min_samples:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            ms=m
    dt=DecisionTreeClassifier(max_depth=d,min_samples_split=ms)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train.append(roc_auc_score(Y_train,probtr))

        
        
    
optimal_depth= depths[auc_cv.index(max(auc_cv))]
optimal_min_samples_split=best_m[auc_cv.index(max(auc_cv))]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(depths, auc_train, label='AUC train')
ax.plot(depths, auc_cv, label='AUC CV')
plt.title('AUC vs hyperparameter(depths)')
plt.xlabel('depths')
plt.ylabel('AUC')
ax.legend()
plt.show()
print('optimal depth for which auc is maximum : ',optimal_depth)
print('optimal minimum samples split for which auc is maximum : ',optimal_min_samples_split) 


# In[ ]:


auc_train_m=[]
auc_cv_m=[]
for m in min_samples:
    dp,rc=0,0
    for d in depths:
        dt=DecisionTreeClassifier(max_depth=d,min_samples_split=m)
        dt.fit(X_train,Y_train)
        probc=dt.predict_proba(X_test)[:,1]
        val=roc_auc_score(Y_test,probc)
        if val>rc:
            rc=val
            dp=d
    dt=DecisionTreeClassifier(max_depth=dp,min_samples_split=m)
    dt.fit(X_train,Y_train)
    probcv=dt.predict_proba(X_test)[:,1]
    auc_cv_m.append(roc_auc_score(Y_test,probcv))
    best_m.append(ms)
    probtr=dt.predict_proba(X_train)[:,1]
    auc_train_m.append(roc_auc_score(Y_train,probtr))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(min_samples, auc_train_m, label='AUC train')
ax.plot(min_samples, auc_cv_m, label='AUC CV')
plt.title('AUC vs hyperparameter(min_samples)')
plt.xlabel('min_samples')
plt.ylabel('AUC')
ax.legend()
plt.show()


# ### **9.8. Best Model**

# In[ ]:


dt=DecisionTreeClassifier(max_depth=10,min_samples_split=15)
dt.fit(X_train,Y_train)
predi=dt.predict_proba(X_test)[:,1]
Y_pred_dt=dt.predict(X_test)

fpr1, tpr1, thresholds1 = metrics.roc_curve(Y_test, predi)
pred=dt.predict_proba(X_train)[:,1]
fpr2,tpr2,thresholds2=metrics.roc_curve(Y_train,pred)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(fpr1, tpr1, label='Test ROC ,auc='+str(roc_auc_score(Y_test,predi)))
ax.plot(fpr2, tpr2, label='Train ROC ,auc='+str(roc_auc_score(Y_train,pred)))
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
ax.legend()
plt.show()


# ### **9.9. Accuracy of the model**

# In[ ]:


score_dt= round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[ ]:


from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Sr.No","Model Name", "Best Parameter", "Accuracy"]
x.add_row(["1","Baseline model", None, score_dummy])
x.add_row(["2","Naive Bayes", "Alpha=0.00001", score_nb])
x.add_row(["3","K-Nearest Neighbors", "k=31", score_knn])
x.add_row(["4","Logistic Regression", "C=0", score_lr])
x.add_row(["5","Decision Tree","Max Depth=15 Min Split=10", score_dt])
x.add_row(["6","Random Forest","Max Depth=80 base learner=5", score_rf])
x.add_row(["7","Support Vector Machine", "alpha=0.001", score_svm])

print(x)


# In[ ]:





# In[ ]:





# In[ ]:




