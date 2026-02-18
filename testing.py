from features import single,main2,main
import pandas as pd
import pickle

def predict(url):
    # single(url)
    url=url.strip()
    if url[0]=='"':
        url=url[1:]
    single(url)
    main('sample.csv','sample2.csv',1)
    df=pd.read_csv('sample2.csv')
    # print(df.head())
    # print(len(df))
    f=open('r_features.pkl','rb')
    relevant_features=pickle.load(f)
    f.close()

    r_f=list(relevant_features.keys())
    df=df[r_f]

    X=df.drop(['phishing'],axis=1).to_numpy()
    f=open('clfs.pkl','rb')
    clfs=pickle.load(f)
    f.close()

    #print(clfs[0].predict(X_test))
    print(X)


    feat=[]

    for i in clfs:
        pred=i.predict(X)
        print(pred)
        feat.append(pred)

    dict1={'clf0':feat[0],'clf1':feat[1],'clf2':feat[2],'clf3':feat[3],'clf4':feat[4],'clf5':feat[5],'clf6':feat[6],'clf7':feat[7],'clf8':feat[8],
    'clf9':feat[9],'label':1}

    dfl=pd.DataFrame(dict1)
    X=dfl.drop(['label'],axis=1)
    f=open('SVM.pkl','rb')
    clf=pickle.load(f)
    f.close()

    pred1=clf.predict(X)[0]
    print(pred1)
    return pred1


    

if __name__=="__main__":
    predict('www.naylorantiques.com/home/HSBC')
    