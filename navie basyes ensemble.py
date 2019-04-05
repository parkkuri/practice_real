# coding: utf-8
 
# In[378]:
 
#���̺꺣���� ���� 
 
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,  HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn.model_selection import train_test_split
 
#������ �������
os.chdir("C:/Users/kebee/Desktop/Data")
train = pd.read_csv('train.csv',sep=',')
 
 
# In[379]:
 
 
#������ ���캸�� 
train.head(6)
 
 
# In[385]:
 
 
#X�� Y�� �������� 
used_features =[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Cabin",
    "Name",
    "Ticket",
    "PassengerId"
    
]
X = train[used_features]
Y = train["Survived"]
 
 
# In[386]:
 
 
#X�� ����ġ�� ������ ���캻�� 
X.isnull().sum()
 
 
# In[387]:
 
 
#Y�� ����ġ�� ���캻�� (����)
Y.isnull().sum()
 
 
# In[388]:
 
 
#Age�� ���� �����ų�, ����ġ�� �����⿡�� �ʹ� �Ʊ�� ������� ��ü�Ѵ� 
train['Age'].fillna(train['Age'].mean(), inplace=True)
 
 
# In[389]:
 
 
#�ʿ���� ���� ������ Cabin�� ����ġ�� �ʹ� ���� �� ��ü�� �����Ѵ� 
del train["PassengerId"]
del train["Cabin"]
del train["Name"]
del train["Ticket"]
 
#�׸��� ���̺꺣�����̹Ƿ� ��� ������͸� ���̺���ȭ�Ѵ�  
train =pd.get_dummies(train)
 
 
# In[390]:
 
 
#Ÿ�ٺ����� ���ȭ���ش� 
train[['Survived']] = train[['Survived']].astype(str)
 
 
# In[391]:
 
 
#������ 
train.head(6)
 
 
# In[392]:
 
 
X = train.loc[:,"Pclass":]
Y = train["Survived"]
 
 
# In[393]:
 
 
#X������ ���� 
X.head(6)
 
 
# In[394]:
 
 
#Y������ ����
Y.head(6)
 
 
# In[395]:
 
 
#����þ� ���� ���� Likelihood �������� ���̺꺣��� ����� 10-fold�� ��Ȯ���� å���Ѵ� 
from  sklearn.naive_bayes  import  GaussianNB 
from  sklearn.model_selection  import  KFold 
from  sklearn.model_selection  import  cross_val_score 
model = GaussianNB()
scores = cross_val_score(model,X.values,Y, cv=10)
scores
 
 
# In[396]:
 
 
#78% ��Ȯ�� 
scores.mean()
 
 
# In[397]:
 
 
#�������� ���� Likelihood ����
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
 
model1 =  BernoulliNB()
scores1 = cross_val_score(model1,X.values,Y, cv=10)
 
scores1
 
 
# In[398]:
 
 
#78% 
scores1.mean()
 
 
# In[399]:
 
 
#���� ���� Likelihood ����
from sklearn.naive_bayes import MultinomialNB
model2 = MultinomialNB()
scores2 = cross_val_score(model2,X.values,Y, cv=10)
 
scores2
 
 
# In[400]:
 
 
#69%�� ���� ������ 
scores2.mean()
 
 
 
 
# coding: utf-8
 
# In[4]:
 
 
# LDA,knn ����
#������ ��� ����
import os
os.chdir("C:/Users/kebee/Desktop/toobig/Data")
 
 
# In[5]:
 
 
#train ������ �������
import pandas as pd
click_train = pd.read_csv('click_train1.txt',sep=',')
profiles_train= pd.read_csv('profiles_train1.txt',sep=',')
 
 
# In[6]:
 
 
# ������ ���캸��
click_train.head(6)
 
 
# In[90]:
 
 
# ������ ���캸��
profiles_train.head(6)
 
 
# In[91]:
 
 
mer_data= pd.merge(click_train, profiles_train, on='id')
mer_data.head(6)
 
 
# In[92]:
 
 
mer_data['time_year']=mer_data.time.astype(str).str[0:4]
 
 
# In[93]:
 
 
mer_data.head(6)
 
 
# In[94]:
 
 
#�� �����ϼ��� ī�����ϴ� �Լ� 
mer_data_time=mer_data.groupby(mer_data.id,as_index=True).count()[["time_year"]]
mer_data_time=mer_data_time.add_suffix('_count').reset_index()
mer_data_time.head(6)
 
 
# In[95]:
 
 
#��ü���ð��� �� ���������� �����̴� 
mer_data_ct=mer_data.groupby(mer_data.id,as_index=True).sum()[["st_c","st_t"]]
mer_data_ct=mer_data_ct.add_suffix('_sum').reset_index()
mer_data_ct.head(6)
 
 
# In[96]:
 
 
#�� 22���� ����Ʈī�װ�(cate)�� �󸶳� �پ��ϰ� �����ߴ����� ���� ������ ����ϱ� ���� �ڵ�
v=mer_data["cate"].groupby(mer_data["id"]).value_counts().unstack("cate")
mer_data_cate_count=v.count(axis=1)
 
 
# In[97]:
 
 
#�� 22���� ����Ʈī�װ�(cate)�� �󸶳� �پ��ϰ� �����ߴ����� ���� ������ ����ϱ� ���� �ڵ�
data_id_cate=pd.DataFrame(mer_data_cate_count.index)
data_id_cate["cate_count"] = mer_data_cate_count.values/22
data_id_cate.head(6)
 
 
# In[98]:
 
 
#ī�װ� ����� �Ͱ� ��������, �������� ����� ���� id�� Ű�� �ΰ� ��ģ�� 
pd1=pd.merge(mer_data_ct,data_id_cate, on='id')
 
 
# In[102]:
 
 
#������ ��ģ �Ͱ� �� �����ϼ� ��Ÿ�� ���� ��ģ�� 
pd2=pd.merge(pd1,mer_data_time, on='id')
pd2.head(10)
 
 
# In[109]:
 
 
#target������ ������ �߰��ϰ� id�� �����Ѵ� 
mer_data_gen=profiles_train.loc[:, ['id','gen']]
pd3=pd.merge(pd2,mer_data_gen, on='id', how='left')
pd3.head(6)
 
 
# In[110]:
 
 
del pd3["id"]
pd3.head(6)
 
 
# In[111]:
 
 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
 
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
 
 
# In[113]:
 
 
pd3.head(6)
 
 
# In[114]:
 
 
#X�� Y�� ������
#Ʈ���̴׼°� �׽�Ʈ �µ� ������ 
X = pd3.loc[:,"st_c_sum":"time_year_count"]
y = pd3.loc[:,"gen"]
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=13903,test_size=0.3)
 
 
# In[115]:
 
 
#Model (QDA, LDA) ���� ����� 
#���� �𵨷� �����Ѵ� 
model1 = QuadraticDiscriminantAnalysis().fit(X_train,y_train)
yhat1 = model1.predict(X_test)
 
 
# In[116]:
 
 
model2 = LinearDiscriminantAnalysis()
yhat2 = model2.fit(X_train,y_train).predict(X_test)
 
 
# In[117]:
 
 
#QDA�� LDA�� confusion matrix
QDA_cnfmatrix2 = confusion_matrix(y_test, yhat1)
 
LDA_cnfmatrix2 = confusion_matrix(y_test, yhat2 )
 
 
# In[118]:
 
 
#��Ȯ�� ����ϴ� �Լ�
def accurary(table): 
    table = pd.DataFrame(table)
    result = (table[0][0]+table[1][1])/(table.sum()[0]+table.sum()[1])
    print(result)
#LDA��Ȯ�� 63% 
accurary(LDA_cnfmatrix2)
 
 
# In[119]:
 
 
#QDA��Ȯ�� 64%
accurary(QDA_cnfmatrix2)
 
 
# In[120]:
 
 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
Qua=  QuadraticDiscriminantAnalysis()
Kfold1=KFold(n_splits=10)
scores1 = cross_val_score(Qua,X,y,cv=Kfold1)
scores1
 
 
# In[121]:
 
 
#QDA 10-ford��� 61%
scores1.mean()
 
 
# In[122]:
 
 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
LDA= LinearDiscriminantAnalysis()
Kfold=KFold(n_splits=10)
scores2 = cross_val_score(LDA,X,y,cv=Kfold)
scores2 
 
 
# In[123]:
 
 
#LDA 10-ford��� 61%
scores2.mean()
 
 
# In[124]:
 
 
#knn���� k�� 3���� ���� ����� ������ ���Ҵ�
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
y_pred3 = clf.fit(X_train, y_train).predict(X_test)
knn_cnfmatrix1 = confusion_matrix(y_test, y_pred3)
 
 
# In[125]:
 
 
pd.DataFrame(knn_cnfmatrix1)
 
 
# In[126]:
 
 
#knn���� k�� 3�϶� ��Ȯ�� 100%, k�� ������ �ʿ�� ����δ�
accurary(knn_cnfmatrix1)
 
 
# In[127]:
 
 
#QDA, LDA, knn, logistic regression �ӻ�� ���� �����
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
 
Qua=  QuadraticDiscriminantAnalysis()
Lda = LinearDiscriminantAnalysis()
knn_en = KNeighborsClassifier()
Logi = LogisticRegression()
 
voting_clf = VotingClassifier(
estimators=[('Lda',Lda),('Qua',Qua),('knn_en',knn_en),('Logi',Logi)], voting='hard')
 
 
 
# In[128]:
 
 
eclf1 = voting_clf.fit(X_train,y_train)
X_test_pre=eclf1.predict(X_test)
 
 
# In[129]:
 
 
eclf1_cnfmatrix2 = confusion_matrix(X_test_pre,y_test)
print(eclf1_cnfmatrix2)
 
 
# In[130]:
 
 
#4�� ��ģ �ӻ�� �� ��Ȯ�� 64%
accurary(eclf1_cnfmatrix2)
 
 
# In[131]:
 
 
#QDA�� ���� �ӻ���� �غ��Ҵ�
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
 
Lda = LinearDiscriminantAnalysis()
knn_en2 = KNeighborsClassifier(n_neighbors=3)
Logi = LogisticRegression()
 
voting_clf2 = VotingClassifier(
estimators=[('Lda',Lda),('knn_en2',knn_en2),('Logi',Logi)], voting='hard')
 
 
 
# In[132]:
 
 
#3�� ��ģ �ӻ�� �� ��Ȯ�� 65%
eclf2 = voting_clf2.fit(X_train,y_train)
X_test_pre2=eclf2.predict(X_test)
eclf1_cnfmatrix3 = confusion_matrix(X_test_pre2,y_test)
accurary(eclf1_cnfmatrix3)
 
 
# In[133]:
 
 
#��� ���� �ӻ���� ���� ����Ű���� �˾ƺ��Ҵ�
for clf in (Lda, knn_en2, Logi):
    clf.fit(X_train, y_train)
    y_pred_ed_1 = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred_ed_1))
 
#knn�� ��Ȯ���� ���� ���� �ٸ��͵��� ��Ȯ���� ����߸��� ���� �� �� �ִ� 
 
 
# In[134]:
 
 
#�𵨸��� �������̰� �����Ƿ� �ܼ� voting�� �ƴ� Ȯ���� ���� ������ ������ִ� soft�ɼ��� �־�Ҵ�
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
 
Lda = LinearDiscriminantAnalysis()
knn_en2 = KNeighborsClassifier()
Logi = LogisticRegression()
 
voting_clf4 = VotingClassifier(
estimators=[('Lda',Lda),('knn_en2',knn_en2),('Logi',Logi)], voting='soft')
 
eclf4 = voting_clf4.fit(X_train,y_train)
 
 
# In[135]:
 
 
#soft�ɼ� �ӻ�� ��Ȯ�� 99%
X_test_prer4=eclf4.predict(X_test)
eclf1_cnfmatrix4 = confusion_matrix(X_test_prer4,y_test)
accurary(eclf1_cnfmatrix4)
 
 
# In[136]:
 
 
eclf1_cnfmatrix4
 
 
# In[137]:
 
 
#test�� ���� ������ �����ϰ� ������ ��ó��
import pandas as pd
click_test = pd.read_csv('click_test1.txt',sep=',')
profiles_test= pd.read_csv('profiles_test1.txt',sep=',')
mer_data_test= pd.merge(click_test, profiles_test, on='id')
mer_data_test_time=mer_data.groupby(mer_data_test.id,as_index=True).count()[["time"]]
mer_data_test_time=mer_data_test_time.add_suffix('_count').reset_index()
mer_data_test= pd.merge(click_test, profiles_test, on='id')
mer_data_ct_test=mer_data_test.groupby(mer_data_test.id,as_index=True).sum()[["st_c","st_t"]]
mer_data_ct_test=mer_data_ct_test.add_suffix('_sum').reset_index()
v_test=mer_data_test["cate"].groupby(mer_data_test["id"]).value_counts().unstack("cate")
mer_data_cate_count_test=v_test.count(axis=1)
data_id_cate_test=pd.DataFrame(mer_data_cate_count_test.index)
data_id_cate_test["cate_count"] = mer_data_cate_count_test.values
pd1_test=pd.merge(mer_data_ct_test,data_id_cate_test, on='id')
pd2_test=pd.merge(pd1_test,mer_data_test_time, on='id')
 
 
# In[138]:
 
 
del pd2_test["id"]
 
 
# In[139]:
 
 
#soft������� ���� �ӻ�� ���� ���� �𵨷� ���ϰ� ��������
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
 
 
X_test_pre_real4=eclf4.predict(pd2_test)
real_final2=pd.concat([pd2_test, pd.DataFrame(X_test_pre_real4)], axis=1)
real_final2.to_csv("�ڱԸ�.csv",encoding="UTF-16",sep='\t')