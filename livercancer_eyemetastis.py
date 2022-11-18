import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
#应用标题
st.set_page_config(page_title='Prediction model for Ocular metastasis of hepatocellular carcinoma')
st.title('Ocular metastasis from primary liver cancer: Machine Learning-Based development and interpretation study')
st.sidebar.markdown('## Variables')
AFP_400 = st.sidebar.selectbox('AFP_400',('≤400','>400'),index=1)
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
hp_train = pd.read_csv('github_data.csv')
hp_train['M'] = hp_train['M'].apply(lambda x : +1 if x==1 else 0)
features =["AFP_400","CEA","CA125","CA199",'ALP','TG']
target = 'M'
random_state_new = 50
data = hp_train[features]
X_data = data
X_ros = np.array(X_data)
y_ros = np.array(hp_train[target])
oversample = SMOTE(random_state = random_state_new)
X_ros, y_ros = oversample.fit_resample(X_ros, y_ros)
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
