import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.svm import SVC
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
test_fields ={"duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"}

trainsRows = [] 
testRows=[]
alpha=0.22;
t=0.00000000005;
A_index=[];
B_index=[];
# reading csv file     
print("1")
trainFilename="KDD Train+.csv";
testFilename="KDD Test+.csv";
train=pd.read_csv("C:\\Users\\rishabh\\Desktop\\KDD Dataset\\"+trainFilename);
test=pd.read_csv("C:\\Users\\rishabh\\Desktop\\KDD Dataset\\"+testFilename);
predictor_Vars=["duration","src_bytes","dst_bytes","logged_in","hot","num_access_files","srv_count","is_guest_login"]
train["label"]=train["label"].fillna("1");
test["label"]=test["label"].fillna("1");
train["num_failed_logins"]=train["num_failed_logins"].fillna("0");
test["num_failed_logins"]=test["num_failed_logins"].fillna("0");
train["num_access_files"]=train["num_access_files"].fillna("0");
test["num_access_files"]=test["num_access_files"].fillna("0");
train["srv_count"]=train["srv_count"].fillna("0");
test["srv_count"]=test["srv_count"].fillna("0");
train["is_guest_login"]=train["is_guest_login"].fillna("0");
test["is_guest_login"]=test["is_guest_login"].fillna("0");
X,Y=train[predictor_Vars],train.label;
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X,Y)
predictions=clf.predict(test[predictor_Vars]);
print(X)
max_feature=0.0;
best_feature="";
for feature in zip(predictor_Vars, clf.feature_importances_):
    print(feature)
    if feature[1]>float(alpha):
        A_index.append(feature[0]);
        if max_feature<feature[1]:
            best_feature=feature[0];
            max_feature=feature[1];
    else:
        B_index.append(feature[0]);
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
sum=0.00;
for xi in B_index:
    predict_a=[str(best_feature)]
    clfA = SVC(kernel='linear',C=1.0,gamma=0.2,cache_size=7000)  
    clfB = SVC(kernel='linear',C=1.0,gamma=0.2,cache_size=7000)  
    print("SVC initialization done");
    clfA.fit(X_train[predictor_Vars],y_train)
    print("SVM on A is done........")
    clfB.fit(X_train[predictor_Vars],y_train)
    print("SVM on B is done........")
    predictA=clfA.predict(X_test[best_feature]);
    predictB=clfB.predict(X_test[best_feature]);
    for feature in zip(best_feature, clfA.feature_importances_):
        print(feature)
        print("\n");
        sum=sum+feature[1];
    for feature in zip(xi, clfB.feature_importances_):
        print(feature)
        sum=sum-feature[1];
    mean_sum=float(sum/len(X_train));
    if mean_sum>t:
        A_index.append(xi); 
print(A_index);   
#predictions1=clf.predict(test[predictor_Vars]);
#print(predictions1)
        
      