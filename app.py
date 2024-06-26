# importing libraries
from datasets import load_dataset, load_dataset_builder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, classification_report
from transformers import Trainer, TrainingArguments
from skops import hub_utils
import pickle
from skops.card import Card, metadata_from_config
from pathlib import Path
from tempfile import mkdtemp, mkstemp
import streamlit as st
from PIL import Image

# Loading the dataset
dataset_name = "saifhmb/social-network-ads"
dataset = load_dataset(dataset_name, split = 'train')
dataset = pd.DataFrame(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the datset into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Logit Reg Model using the Training set
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting the Test result
y_pred = model.predict(X_test)

# Making the Confusion Matrix and evaluating performance
cm = confusion_matrix(y_pred, y_test, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
acc = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)

# Pickling the model
pickle_out = open("model.pkl", "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()

# Loading the model to predict on the data
pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in) 

def welcome(): 
    return 'welcome all'

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Age, EstimatedSalary):
    prediction = model.predict(sc.transform([[Age, EstimatedSalary]]))
    print(prediction)
    return prediction

# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    st.title("Customer Purchase Prediction ML App") 
    st.header("Model Description", divider = "gray")
    multi = '''This is a logistic regression classifier trained on social network ads dataset to predict customer purchasing behavior
    based on their Age and Estimated Salary.
    For more details on the model please refer to the model card at https://huggingface.co/saifhmb/social-network-ads-logit-model
    '''
    st.markdown(multi)
    st.header("Model Metrics", divider = "gray")
    st.metric(label = "Accuracy", value = round(acc, 2))
    st.metric(label = "Precision", value = round(ps, 2))
    st.metric(label = "Recall", value = round(rs, 2))

    st.markdown("To determine whether a customer will make a purchase, please **enter** the Age and Estimated Salary:")
    Age = st.number_input("Age")
    EstimatedSalary = st.number_input("Estimated Salary")
    result = ""
    if st.button("Predict"):
        result = prediction(Age, EstimatedSalary)
        if result == 0:
            st.success("The output is {}".format(result) + " and thus the customer is NOT likely to complete a purchase")
        if result == 1:
            st.success("The output is {}".format(result) + " and thus the customer is likely to complete a purchase")


        
if __name__=='__main__': 
    main() 
