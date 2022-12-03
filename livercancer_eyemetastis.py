import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
# from sklearn.neural_network import MLPClassifier
# from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBClassifier
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#应用标题
#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Prediction model for Ocular metastasis of hepatocellular carcinoma')
st.title('Ocular metastasis from primary liver cancer: Machine Learning-Based development and interpretation study')
st.sidebar.markdown('## Variables')
AFP_400 = st.sidebar.selectbox('AFP_400(μg/L)',('≤400','>400'),index=1)
CEA = st.sidebar.slider("CEA(μg/L)", 0.00, 120.00, value=7.68, step=0.01)
CA125 = st.sidebar.slider("CA125(μg/L)", 0.00, 500.00, value=30.00, step=0.01)
CA199 = st.sidebar.slider("CA199(μg/L)", 0.00, 500.00, value=59.61, step=0.01)
ALP = st.sidebar.slider("ALP(U/L)", 0, 1000, value=215, step=1)
TG = st.sidebar.slider("TG(mmol/L)", 0.00,10.00, value=1.42, step=0.01)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'≤400':0,'>400':1}
AFP_400 =map[AFP_400]
# 数据读取，特征标注
#%%load model
import joblib
xgb_model = joblib.load('gbm_model_liver_eye.pkl')
hp_train = pd.read_csv('github_data.csv')
features =["AFP_400","CEA","CA125","CA199",'ALP','TG']
target = 'M'
y = np.array(hp_train[target])
sp = 0.5
#figure
is_t = (xgb_model.predict_proba(np.array([[AFP_400,CEA,CA125,CA199,ALP,TG]]))[0][1])> sp
prob = (xgb_model.predict_proba(np.array([[AFP_400,CEA,CA125,CA199,ALP,TG]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Ocular metastasis'
else:
    result = 'Low Risk Ocular metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Ocular metastasis group:  '+str(prob)+'%')

    #%%SHAP
    col_names = features
    X_last = pd.DataFrame(np.array([[AFP_400,CEA,CA125,CA199,ALP,TG]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0
    
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
        
    
    model = xgb_model
    #%%
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    # st.subheader('SHAP Force logit plot')
    # fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    # force_plot = shap.force_plot(explainer.expected_value,
    #                 shap_values[a, :], 
    #                 X.iloc[a, :], 
    #                 figsize=(25, 3),
    #                 # link = "logit",
    #                 matplotlib=True,
    #                 out_names = "Output value")
    # st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix
    st.subheader('Confusion Matrix of XGB')
    xgb_prob = xgb_model.predict(X)
    cm = confusion_matrix(y, xgb_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NOM', 'OM'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of XGB")
    disp1 = plt.show()
    st.pyplot(disp1)

