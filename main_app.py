#importing the necessary libraries for the work

import streamlit as st #for deploying work
import numpy as np 
import pandas as pd 
from PIL import Image # to display the illustration
import subprocess #descriptor calculation
import os #for file handling
import base64 #for encoding and deconding the file
import pickle #loading pkl model
import io

# now we will be defining three functions in total

#first function is basically to generate the descriptors output file from the input file

# so calculating the molecular descriptor

def descriptor_calc():
    # so basically here we are performing the descriptor calculations
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar C:/Users/shash/Desktop/Projects/CDD_project/padel/padel/PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes C:/Users/shash/Desktop/Projects/CDD_project/padel/padel/PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process=subprocess.Popen(bashCommand.split(),stdout=subprocess.PIPE)
    output,error=process.communicate()
    os.remove('molecule.smi')
    

# now below function is defined for downloading the file
def download_file(df):
    csv = df.to_csv(index=False, encoding='utf-8', na_rep='')
    encoded_csv = csv.encode('utf-8', 'ignore')  # Ignore any characters that cannot be encoded
    b64 = base64.b64encode(encoded_csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href



# below function is defined for model building
def model_build(input_data):

    #firstly we have to load the saved model and give the input data into it to get the predictions
    load_model = pickle.load(open('acetylcholinesterase_model.pkl', 'rb'), encoding='latin1')
    predictions=load_model.predict(input_data)

    st.header('**Prediction output**')
    prediction_output=pd.Series(predictions,name='pIC50')
    molecule_name=pd.Series(load_data[1],name='molecule_name')
    df=pd.concat([molecule_name,prediction_output],axis=1)
    st.write(df)
    st.markdown(download_file(df),unsafe_allow_html=True)

#loading the logo image
image=Image.open('logo.png')
st.image(image,use_column_width=True) #for loading the image to webapp

#now description part of the page
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app allows you to predict the bioactivity towards inhibting the `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for Alzheimer's disease.

**Credits**
- App built in `Python` + `Streamlit` by Shashank Asthana with reference from [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).
---
"""
)

# setting  up the side bar for loading the input molecule csv file
with st.sidebar.header('1. Upload your CSV data :'):
    uploaded_file=st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button('Predict'):
    load_data=pd.read_table(uploaded_file,sep=' ',header=None)
    load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

    st.header('**Original Input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        descriptor_calc()
    
    #reading the calculated descriptors and displaying the dataframe
    st.header('**Calculated molecular descriptors**')
    read_desc=pd.read_csv('descriptors_output.csv')
    st.write(read_desc)
    st.write(read_desc.shape)

    st.header("**Subset of descriptors from previously built models**")
    X_list=list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset=read_desc[X_list]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # now applying the model

    model_build(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')




