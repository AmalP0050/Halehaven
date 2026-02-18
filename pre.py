import pandas as pd 

df1=pd.read_csv('DeepPhishURLS.csv')
df2=pd.read_csv('ThreatActor1URLS_p.csv')
df3=pd.read_csv('urldata.csv')

print(df1.columns)
print(df2.columns)
print(df3.columns)

print(len(df3))
df3=df3[df3.label=='good']



df3['label']=[0 for i in df3.url]


df3=df3[['url','label']][:2000]
print(df3.head())
print(len(df3))

def rem(x):
    x=x[7:]
    #print(x)
    return x

df1['url']=[i for i in df1.URL]
df1['label']=[1 for i in df1.url]
df1=df1[['url','label']]
df1.url=df1.url.apply(rem)
print(df1.head())
print(len(df1))





df2['url']=[i for i in df2.URL]
df2['label']=[1 for i in df2.url]
df2=df2[['url','label']]
df2.url=df2.url.apply(rem)
print(df2.head())
print(len(df2))

from features import main

dfy=pd.concat([df1, df2])['url'][:1000]
df3=df3['url']

df3.to_csv('nophish.csv',index=False,header=False)
dfy.to_csv('phish.csv',index=False,header=False)

# main('nophish.csv','result1.csv',0)
#main('phish.csv','result2.csv',1)

dfn=pd.read_csv('result1.csv')
dfy=pd.read_csv('result2.csv')

dtypes1=list(dict(dfn.dtypes).keys())
dtypes2=list(dict(dfn.dtypes).values())

print(dtypes2[0])
print(type(dtypes2[0]))
drop_list=[]
c_list=[]
for i in range(len(dtypes2)):
    if str(dtypes2[i])!='int64':
        print(str(dtypes1[i]))

        if str(dtypes2[i])=='bool':
            c_list.append(dtypes1[i])
        else: 
            drop_list.append(dtypes1[i])
       

dfn=dfn.drop(drop_list,axis=1)
dfy=dfy.drop(drop_list,axis=1)

df=pd.concat([dfy, dfn])

for i in range(len(c_list)):
    df[c_list[i]]=[int(j) for j in df[c_list[i]]]

print(df.ip_exist.head())

corr=df.corr()
cor_target = abs(corr["phishing"])
#Selecting highly correlated features
relevant_features = dict(cor_target[cor_target>0.3])
print(relevant_features)

import pickle

f=open('r_features.pkl','wb')
pickle.dump(relevant_features,f)
f.close()

r_f=list(relevant_features.keys())
df=df[r_f]

print(df.head())

X=df.drop(['phishing'],axis=1)
y=df['phishing']

from sklearn.ensemble import AdaBoostClassifier
clf1 = AdaBoostClassifier(n_estimators=100, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

from sklearn import tree
clf3 = tree.DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(max_depth=2, random_state=0)

from sklearn.ensemble import ExtraTreesClassifier
clf5 = ExtraTreesClassifier(n_estimators=100, random_state=0)

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
clf6 = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf7 = KNeighborsClassifier(n_neighbors=3)

from sklearn.linear_model import LogisticRegression
clf8 = LogisticRegression(random_state=0)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
clf9 = make_pipeline(StandardScaler(), SVC(gamma='auto'))

from sklearn.neural_network import MLPClassifier
clf10 = MLPClassifier(random_state=1, max_iter=300)

clf_list=[clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
X_train=X_train.to_numpy()
X_test=X_test.to_numpy()

print(X_train[0])
#n/0

def fit_parallel(x):
    x.fit(X_train,y_train)

    return x

from joblib import Parallel, delayed
clfs=Parallel(n_jobs=4, max_nbytes=None)(delayed(fit_parallel)(i) for i in clf_list)
print(clfs)

f=open('clfs.pkl','wb')
pickle.dump(clfs,f)
f.close()

#print(clfs[0].predict(X_test))

feat=[]
from sklearn.metrics import accuracy_score
for i in clfs:
    pred=i.predict(X_train)
    ac=accuracy_score(pred,y_train)
    print(ac)
    feat.append(pred)

dict1={'clf0':feat[0],'clf1':feat[1],'clf2':feat[2],'clf3':feat[3],'clf4':feat[4],'clf5':feat[5],'clf6':feat[6],'clf7':feat[7],'clf8':feat[8],
'clf9':feat[9],'label':y_train}

dfl=pd.DataFrame(dict1)
X=dfl.drop(['label'],axis=1)
y=dfl['label']

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X,y)

f=open('SVM.pkl','wb')
pickle.dump(clf,f)
f.close()

pred1=clf.predict(X[:200])
print(pred1)
ac=accuracy_score(y[:200],pred1)
print(ac)

f=open('accuracy.pkl','wb')
pickle.dump(ac,f)
f.close()

























