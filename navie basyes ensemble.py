# coding: utf-8
 
# In[378]:
 
#나이브베이즈 과제 
 
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,  HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn.model_selection import train_test_split
 
#데이터 끌어오기
os.chdir("C:/Users/kebee/Desktop/Data")
train = pd.read_csv('train.csv',sep=',')
 
 
# In[379]:
 
 
#데이터 살펴보기 
train.head(6)
 
 
# In[385]:
 
 
#X와 Y를 나눠본다 
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
 
 
#X의 결측치를 열마다 살펴본다 
X.isnull().sum()
 
 
# In[387]:
 
 
#Y의 결측치를 살펴본다 (없다)
Y.isnull().sum()
 
 
# In[388]:
 
 
#Age의 열을 버리거나, 결측치를 버리기에는 너무 아까워 평균으로 대체한다 
train['Age'].fillna(train['Age'].mean(), inplace=True)
 
 
# In[389]:
 
 
#필요없는 열을 버리고 Cabin은 결측치가 너무 많아 열 자체를 제거한다 
del train["PassengerId"]
del train["Cabin"]
del train["Name"]
del train["Ticket"]
 
#그리고 나이브베이즈이므로 모든 명목데이터를 더미변수화한다  
train =pd.get_dummies(train)
 
 
# In[390]:
 
 
#타겟변수를 명목화해준다 
train[['Survived']] = train[['Survived']].astype(str)
 
 
# In[391]:
 
 
#데이터 
train.head(6)
 
 
# In[392]:
 
 
X = train.loc[:,"Pclass":]
Y = train["Survived"]
 
 
# In[393]:
 
 
#X데이터 최종 
X.head(6)
 
 
# In[394]:
 
 
#Y데이터 최종
Y.head(6)
 
 
# In[395]:
 
 
#가우시안 정규 분포 Likelihood 모형으로 나이브베이즈를 만들고 10-fold로 정확도를 책정한다 
from  sklearn.naive_bayes  import  GaussianNB 
from  sklearn.model_selection  import  KFold 
from  sklearn.model_selection  import  cross_val_score 
model = GaussianNB()
scores = cross_val_score(model,X.values,Y, cv=10)
scores
 
 
# In[396]:
 
 
#78% 정확도 
scores.mean()
 
 
# In[397]:
 
 
#베르누이 분포 Likelihood 모형
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
 
 
#다항 분포 Likelihood 모형
from sklearn.naive_bayes import MultinomialNB
model2 = MultinomialNB()
scores2 = cross_val_score(model2,X.values,Y, cv=10)
 
scores2
 
 
# In[400]:
 
 
#69%로 제일 안좋다 
scores2.mean()
 
 
 
 
# coding: utf-8
 
# In[4]:
 
 
# LDA,knn 과제
#데이터 경로 지정
import os
os.chdir("C:/Users/kebee/Desktop/toobig/Data")
 
 
# In[5]:
 
 
#train 데이터 끌어오기
import pandas as pd
click_train = pd.read_csv('click_train1.txt',sep=',')
profiles_train= pd.read_csv('profiles_train1.txt',sep=',')
 
 
# In[6]:
 
 
# 데이터 살펴보기
click_train.head(6)
 
 
# In[90]:
 
 
# 데이터 살펴보기
profiles_train.head(6)
 
 
# In[91]:
 
 
mer_data= pd.merge(click_train, profiles_train, on='id')
mer_data.head(6)
 
 
# In[92]:
 
 
mer_data['time_year']=mer_data.time.astype(str).str[0:4]
 
 
# In[93]:
 
 
mer_data.head(6)
 
 
# In[94]:
 
 
#총 접속일수를 카운팅하는 함수 
mer_data_time=mer_data.groupby(mer_data.id,as_index=True).count()[["time_year"]]
mer_data_time=mer_data_time.add_suffix('_count').reset_index()
mer_data_time.head(6)
 
 
# In[95]:
 
 
#총체류시간과 총 페이지뷰의 총합이다 
mer_data_ct=mer_data.groupby(mer_data.id,as_index=True).sum()[["st_c","st_t"]]
mer_data_ct=mer_data_ct.add_suffix('_sum').reset_index()
mer_data_ct.head(6)
 
 
# In[96]:
 
 
#총 22개의 사이트카테고리(cate)에 얼마나 다양하게 접속했는지에 대한 비율을 계산하기 위한 코드
v=mer_data["cate"].groupby(mer_data["id"]).value_counts().unstack("cate")
mer_data_cate_count=v.count(axis=1)
 
 
# In[97]:
 
 
#총 22개의 사이트카테고리(cate)에 얼마나 다양하게 접속했는지에 대한 비율을 계산하기 위한 코드
data_id_cate=pd.DataFrame(mer_data_cate_count.index)
data_id_cate["cate_count"] = mer_data_cate_count.values/22
data_id_cate.head(6)
 
 
# In[98]:
 
 
#카테고리 계산한 것과 페이지수, 페이지뷰 계산한 열을 id를 키로 두고 합친다 
pd1=pd.merge(mer_data_ct,data_id_cate, on='id')
 
 
# In[102]:
 
 
#위에서 합친 것과 총 접속일수 나타난 열을 합친다 
pd2=pd.merge(pd1,mer_data_time, on='id')
pd2.head(10)
 
 
# In[109]:
 
 
#target변수인 성별을 추가하고 id를 제거한다 
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
 
 
#X와 Y를 나눈다
#트레이닝셋과 테스트 셋도 나눈다 
X = pd3.loc[:,"st_c_sum":"time_year_count"]
y = pd3.loc[:,"gen"]
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=13903,test_size=0.3)
 
 
# In[115]:
 
 
#Model (QDA, LDA) 모델을 만든다 
#만든 모델로 예측한다 
model1 = QuadraticDiscriminantAnalysis().fit(X_train,y_train)
yhat1 = model1.predict(X_test)
 
 
# In[116]:
 
 
model2 = LinearDiscriminantAnalysis()
yhat2 = model2.fit(X_train,y_train).predict(X_test)
 
 
# In[117]:
 
 
#QDA와 LDA의 confusion matrix
QDA_cnfmatrix2 = confusion_matrix(y_test, yhat1)
 
LDA_cnfmatrix2 = confusion_matrix(y_test, yhat2 )
 
 
# In[118]:
 
 
#정확도 계산하는 함수
def accurary(table): 
    table = pd.DataFrame(table)
    result = (table[0][0]+table[1][1])/(table.sum()[0]+table.sum()[1])
    print(result)
#LDA정확도 63% 
accurary(LDA_cnfmatrix2)
 
 
# In[119]:
 
 
#QDA정확도 64%
accurary(QDA_cnfmatrix2)
 
 
# In[120]:
 
 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
Qua=  QuadraticDiscriminantAnalysis()
Kfold1=KFold(n_splits=10)
scores1 = cross_val_score(Qua,X,y,cv=Kfold1)
scores1
 
 
# In[121]:
 
 
#QDA 10-ford결과 61%
scores1.mean()
 
 
# In[122]:
 
 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
LDA= LinearDiscriminantAnalysis()
Kfold=KFold(n_splits=10)
scores2 = cross_val_score(LDA,X,y,cv=Kfold)
scores2 
 
 
# In[123]:
 
 
#LDA 10-ford결과 61%
scores2.mean()
 
 
# In[124]:
 
 
#knn에서 k를 3으로 모델을 만들어 예측해 보았다
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
y_pred3 = clf.fit(X_train, y_train).predict(X_test)
knn_cnfmatrix1 = confusion_matrix(y_test, y_pred3)
 
 
# In[125]:
 
 
pd.DataFrame(knn_cnfmatrix1)
 
 
# In[126]:
 
 
#knn에서 k가 3일때 정확도 100%, k를 변경할 필요는 없어보인다
accurary(knn_cnfmatrix1)
 
 
# In[127]:
 
 
#QDA, LDA, knn, logistic regression 앙상블 모델을 만든다
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
 
 
#4개 합친 앙상블 모델 정확도 64%
accurary(eclf1_cnfmatrix2)
 
 
# In[131]:
 
 
#QDA를 빼고 앙상블을 해보았다
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
 
Lda = LinearDiscriminantAnalysis()
knn_en2 = KNeighborsClassifier(n_neighbors=3)
Logi = LogisticRegression()
 
voting_clf2 = VotingClassifier(
estimators=[('Lda',Lda),('knn_en2',knn_en2),('Logi',Logi)], voting='hard')
 
 
 
# In[132]:
 
 
#3개 합친 앙상블 모델 정확도 65%
eclf2 = voting_clf2.fit(X_train,y_train)
X_test_pre2=eclf2.predict(X_test)
eclf1_cnfmatrix3 = confusion_matrix(X_test_pre2,y_test)
accurary(eclf1_cnfmatrix3)
 
 
# In[133]:
 
 
#어떠한 모델이 앙상블의 모델을 향상시키는지 알아보았다
for clf in (Lda, knn_en2, Logi):
    clf.fit(X_train, y_train)
    y_pred_ed_1 = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred_ed_1))
 
#knn이 정확도가 제일 높고 다른것들이 정확도를 떨어뜨리는 것을 볼 수 있다 
 
 
# In[134]:
 
 
#모델마다 성능차이가 있으므로 단순 voting이 아닌 확률이 높은 쪽으로 계산해주는 soft옵션을 넣어보았다
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
 
Lda = LinearDiscriminantAnalysis()
knn_en2 = KNeighborsClassifier()
Logi = LogisticRegression()
 
voting_clf4 = VotingClassifier(
estimators=[('Lda',Lda),('knn_en2',knn_en2),('Logi',Logi)], voting='soft')
 
eclf4 = voting_clf4.fit(X_train,y_train)
 
 
# In[135]:
 
 
#soft옵션 앙상블 정확도 99%
X_test_prer4=eclf4.predict(X_test)
eclf1_cnfmatrix4 = confusion_matrix(X_test_prer4,y_test)
accurary(eclf1_cnfmatrix4)
 
 
# In[136]:
 
 
eclf1_cnfmatrix4
 
 
# In[137]:
 
 
#test셋 위의 과정과 동일하게 데이터 전처리
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
 
 
#soft방식으로 만든 앙상블 모델을 최종 모델로 택하고 내보낸다
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
 
 
X_test_pre_real4=eclf4.predict(pd2_test)
real_final2=pd.concat([pd2_test, pd.DataFrame(X_test_pre_real4)], axis=1)
real_final2.to_csv("박규리.csv",encoding="UTF-16",sep='\t')