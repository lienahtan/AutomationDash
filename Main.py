import pandas as pd
import numpy as np  
from Module import module

import streamlit as st
from typing import Dict, Callable
from Overall import overall
from PIL import Image


st.set_page_config(
    page_title="PE Department Analysis",
    page_icon="📊",
    layout="wide",
)      


image = Image.open(r"C:\Users\SPLSIP149\Downloads\Shimano.jpeg")
st.sidebar.image(image)

# side bar
st.sidebar.title("Welcome to PE Department!")

st.sidebar.markdown(f'<h1 style="color:#1974D2	; font-size:18px;">{"Step 1:"}</h1>',
                    unsafe_allow_html=True)
# choose which automation to view, goes to the sheet of the data
automation = st.sidebar.selectbox(
    label="Choose automation",
    options=[
        "Machining",
        "CF 655",
    ]
)

# choose which level to view
lens = st.sidebar.selectbox(
    label="Choose your lens",
    options=[
        "Automation Overview",
        "Individual Module Overview",
    ]
)

# mapping buttons to 
page_function_mapping: Dict[str, Callable[[pd.DataFrame], None]] = {
    ("Automation Overview"): overall,
    ("Individual Module Overview"): module,
   
}

st.sidebar.markdown(f'<h1 style="color:#1974D2	; font-size:18px;">{"Step 2:"}</h1>',
                    unsafe_allow_html=True)
# Button to upload new data
st.sidebar.subheader("Upload your EXCEL Data here")
data_file = st.sidebar.file_uploader('Please ensure it is in the appropriate format')

if data_file is not None:
    df = pd.read_excel(data_file, sheet_name= automation)
    df = df.iloc[:, : -3]
else:
    # Nathaniel's data set location
    # TO BE USED FOR EASE OF CHANGING OF CODE
    dataset_location = r"C:\Users\SPLSIP149\Desktop\Data from Adrian - Newest\Tester.xlsx"
    df = pd.read_excel(dataset_location, sheet_name= automation).iloc[:, : -3]
    st.warning("Please upload an excel file if not you are running a TESTER file.")
   
   
st.sidebar.markdown(f'<h1 style="color:#1974D2; font-size:18px;">{"Step 3:"}</h1>',
                    unsafe_allow_html=True)
#Selection of Data range  
#Auto put in earliest and latest date in data given

start = st.sidebar.date_input("Select your date range.",[min(df['ProdnDate']), max(df['ProdnDate'])])
startDate = start[0]
endDate = start[1]

st.sidebar.markdown(f'<h5 style="color:#C32148; font-size:13px;">{"Alter ONLY if you need a custom date range."}</h1>',
                    unsafe_allow_html=True)


page_function_mapping[lens](df, automation, startDate, endDate)
