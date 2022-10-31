import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
#应用标题
st.set_page_config(page_title='Prediction model for Ocular metastasis of hepatocellular carcinoma')
st.title('Ocular metastasis from primary liver cancer: Machine Learning-Based development and interpretation study')
st.sidebar.markdown('## Variables')
AFP_400 = st.sidebar.selectbox('AFP_400',('≤400','>400'),index=1)
# CEA = st.sidebar.selectbox('CEA',('Single tumor','Multiple tumor'),index=0)
# vascular_invasion = st.sidebar.selectbox('vascular_invasion',('No','Yes'),index=1)
# BCLC = st.sidebar.selectbox('BCLC',('Stage 0','Stage A','Stage B','Stage C'),index=1)
CEA = st.sidebar.slider("CEA", 0.00, 400.00, value=7.68, step=0.01)
CA125 = st.sidebar.slider("CA125", 0.00, 5000.00, value=3320.00, step=0.01)
CA199 = st.sidebar.slider("CA199", 0.00, 2000.00, value=59.61, step=0.01)
ALP = st.sidebar.slider("ALP", 0, 2000, value=215, step=1)
TG = st.sidebar.slider("TG", 0.00,10.00, value=1.42, step=0.01)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'≤400':0,'>400':1}
AFP_400 =map[AFP_400]
# 数据读取，特征标注
hp_train = pd.read_csv('E:\\Spyder_2022.3.29\\output\\machinel\\sy_output\\liver_cacer_em\\github_data.csv')
hp_train['M'] = hp_train['M'].apply(lambda x : +1 if x==1 else 0)
features =["AFP_400","CEA","CA125","CA199",'ALP','TG']
target = 'M'
random_state_new = 50
data = hp_train[features]
for name in ['CEA','CA125','ALP']:
    X = data.drop(columns=f"{name}")
    Y = data.loc[:, f"{name}"]
    X_0 = SimpleImputer(missing_values=np.nan, strategy="constant").fit_transform(X)
    y_train = Y[Y.notnull()]
    y_test = Y[Y.isnull()]
    x_train = X_0[y_train.index, :]
    x_test = X_0[y_test.index, :]

    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)

    data.loc[Y.isnull(), f"{name}"] = y_predict
    
X_data = data

X_ros = np.array(X_data)
# X_ros = np.array(hp_train[features])
y_ros = np.array(hp_train[target])
# mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
#                     alpha=0.0001,
#                     batch_size='auto',
#                     learning_rate='constant',
#                     learning_rate_init=0.01,
#                     power_t=0.5,
#                     max_iter=200,
#                     shuffle=True, random_state=random_state_new)
XGB_model = XGBClassifier(n_estimators=360, max_depth=2, learning_rate=0.1,random_state = random_state_new)
XGB_model.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (XGB_model.predict_proba(np.array([[AFP_400,CEA,CA125,CA199,ALP,TG]]))[0][1])> sp
prob = (XGB_model.predict_proba(np.array([[AFP_400,CEA,CA125,CA199,ALP,TG]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Ocular metastasis'
else:
    result = 'Low Risk Ocular metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Ocular metastasis group:  '+str(prob)+'%')