import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import keras
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from keras.utils import pad_sequences
from sklearn import preprocessing

## Loading the ann model
### Opening a file with Pickle (Logistic Regression model was saved as Pickle (binary) format)
with open('Poisonous_Mushroom_model.pkl', 'rb') as file:
          model = pickle.load(file)

## load the copy of the dataset
data_csv = pd.read_csv('mushrooms.csv')

## set page configuration
st.set_page_config(page_title= 'Poisonous Mushroom', layout='wide')

## add page title and content
st.title('Identifying Poisonous Mushroom using Logistic Regression')
st.write('''
 Poisonous Mushroom Prediction App.

 This app predicts if a mushroom is poisonous or not by some of their features.
''')

#st.sidebar.header('User Input Features')

## add image
image = Image.open('images\mushroom.jpg')
st.image(image, width=800)


## get user imput
#email_text = st.text_input('Email Text:')
st.sidebar.header('User Input Features')
def user_input_features():
    capshape = st.sidebar.selectbox("cap-shape:", ('b','c','f','k','s','x'))
    capsurface = st.sidebar.selectbox("cap-surface:",('f','g','s','y'))
    capcolor = st.sidebar.selectbox("cap-color:",('b','c','e','g','n','p','r','u','w','y'))
    bruises = st.sidebar.selectbox("bruises:",('f','t'))
    odor = st.sidebar.selectbox("odor:",('a','c','f','l','m','n','p','s','y'))
    gillattachment = st.sidebar.selectbox("gill-attachment:",('a','f'))
    gillspacing = st.sidebar.selectbox("gill-spacing:",('c','w'))
    gillsize = st.sidebar.selectbox("gill-size:",('n','b'))
    gillcolor = st.sidebar.selectbox("gill-color:",('b','e','g','h','k','n','o','p','r','u','w','y'))
    stalkshape = st.sidebar.selectbox("stalk-shape:",('e','t'))
    stalkroot = st.sidebar.selectbox("stalk-root:",('b','c','e','r'))
    stalksurfaceabovering = st.sidebar.selectbox("stalk-surface-above-ring:",('f','k','s','y'))
    stalksurfacebelowring = st.sidebar.selectbox("stalk-surface-below-ring:",('f','k','s','y'))
    stalkcolorabovering = st.sidebar.selectbox("stalk-color-above-ring:",('b','c','e','g','n','o','p','w','y'))
    stalkcolorbelowring = st.sidebar.selectbox("stalk-color-below-ring:",('b','c','e','g','n','o','p','w','y'))
    veiltype = st.sidebar.selectbox("veil-type:",('p'))
    veilcolor = st.sidebar.selectbox("veil-color:",('n','o','w','y'))
    ringnumber = st.sidebar.selectbox("ring-number:",('n','o','t'))
    ringtype = st.sidebar.selectbox("ring-type:",('e','f','l','n','p'))
    sporeprintcolor = st.sidebar.selectbox("spore-print-color:",('b','h','k','n','o','r','u','w','y'))
    population = st.sidebar.selectbox("population:",('a','c','n','s','v','y'))
    habitat = st.sidebar.selectbox("habitat:",('d','g','l','m','p','u','w'))


    data = {'cap-shape':capshape,
            'cap-surface':capsurface,
            'cap-color':capcolor,
            'bruises':bruises,
            'odor':odor,
            'gill-attachment':gillattachment,
            'gill-spacing':gillspacing,
            'gill-size':gillsize,
            'gill-color':gillcolor,
            'stalk-shape':stalkshape,
            'stalk-root':stalkroot,
            'stalk-surface-above-ring':stalksurfaceabovering,
            'stalk-surface-below-ring':stalksurfacebelowring,
            'stalk-color-above-ring':stalkcolorabovering,
            'stalk-color-below-ring':stalkcolorbelowring,
            'veil-type':veiltype,
            'veil-color':veilcolor,
            'ring-number':ringnumber,
            'ring-type':ringtype,
            'spore-print-color':sporeprintcolor,
            'population':population,
            'habitat':habitat
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.write(input_df.iloc[:,:12])
st.write(input_df.iloc[:,12:])
## Combines user input with entire dataset
data_csv = data_csv.drop(columns=['class'])
df = pd.concat([input_df,data_csv],axis=0)

## Replacing missing values
## I'm replacing the "?" with "b" in stalk-root column
df = df.replace(to_replace= "?", value = "b")

# Encoding 
encoder = preprocessing.LabelEncoder() # creating labelEncoder
# Converting strings into numbers
# 1) Binomial features
for column in df.columns:
    if len(df[column].unique()) ==2:
        df[column] = encoder.fit_transform(df[column])
    else:
        dummies = pd.get_dummies(df[column],prefix=column)
        df = pd.concat((df, dummies), axis = 1)
        df.drop(column, axis = 1, inplace = True)

# 2) Rest of features
# for column in df.columns:
#     if len(df[column].unique()) != 2:
#         dummies = pd.get_dummies(df[column],prefix=column)
#         df = pd.concat((df, dummies), axis = 1)
#         df.drop(column, axis = 1, inplace = True)

df2 = df[:1]

#st.write(df2)

# Make prediction
prediction = model.predict(df2)

st.subheader('Prediction')
st.write(prediction)

if prediction > 0.5:
    st.write('This mushroom is likely to to be poisonous')
else:
    st.write('This mushroom is likely to be edible')

#Finally, in the Terminal, I have to write: python -m streamlit run PoisonousMushroom_app.py
