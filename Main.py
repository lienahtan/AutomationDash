import pandas as pd
import numpy as np  
from Operator import operator
from Predictions import predicting
import streamlit as st
from typing import Dict, Callable
from Overall import overall
from PIL import Image
import datetime


st.set_page_config(
    page_title="PE Department Analysis",
    page_icon="📊",
    layout="wide",
)      

image = Image.open(r"AAA.png")
st.image(image, width=480)

image = Image.open(r"Shimano.jpeg")
st.sidebar.image(image)

rerun = st.sidebar.button('Rerun')
if rerun:
    st.experimental_rerun()
    
st.sidebar.info("[Guide to AAA's Full Functionality.](https://docs.google.com/spreadsheets/d/1-ZC0yg14K4Yh2BwuT1ZGfgvPtpxbr1sAr2lYWbfurr0/edit?usp=sharing)")

# choose which automation to view, goes to the sheet of the data
st.sidebar.markdown(f'<h1 style="color:#1974D2	; font-size:18px;">{"Step 1:"}</h1>',
                    unsafe_allow_html=True)

excelSheet = pd.ExcelFile(r"TestData.xlsx")
automation = st.sidebar.selectbox(
    label="Choose automation",
    options=excelSheet.sheet_names
)

# choose which level to view
lens = st.sidebar.selectbox(
    label="Choose your lens",
    options=[
        "Automation Overview",
        "Operator Insight",
    ]
)

# mapping buttons to 
page_function_mapping: Dict[str, Callable[[pd.DataFrame], None]] = {
    ("Automation Overview"): overall,
    ("Operator Insight"): operator,
    ("Predictions"): predicting,
   
}

# Button to upload new data
st.sidebar.markdown(f'<h1 style="color:#1974D2	; font-size:18px;">{"Step 2:"}</h1>',
                    unsafe_allow_html=True)

data_file = st.sidebar.file_uploader('Please ensure it is in the appropriate format')

if data_file is not None:
    df = pd.read_excel(data_file, sheet_name= automation)
    df = df.iloc[:, : ]
else:
    # Nathaniel's data set location
    # TO BE USED FOR EASE OF CHANGING OF CODE
    st.sidebar.warning("Please upload an excel file if not you are running a TESTER file.")
    dataset_location = r"TestData.xlsx"
    df = pd.read_excel(dataset_location, sheet_name= automation).iloc[:, : ]
   
#Selection of Data range for primary analysis
#Auto put in earliest and latest date in data given
st.sidebar.markdown(f'<h1 style="color:#1974D2; font-size:18px;">{"Step 3:"}</h1>',
                    unsafe_allow_html=True)

start = st.sidebar.date_input("Select your date range.",[min(df['ProdnDate']), max(df['ProdnDate'])])
startDate = start[0]
endDate = start[1]
# startDate = '2022-03-01'
# endDate = '2022-05-30'

st.sidebar.markdown(f'<h5 style="color:#C32148; font-size:13px;">{"Alter ONLY if you need a custom date range."}</h1>',
                    unsafe_allow_html=True)

# Getting date range that is used to make comparison
# If no date is given, default is on month before the start of the date range in Step 3
st.sidebar.markdown(f'<h1 style="color:#1974D2; font-size:18px;">{"Step 4:"}</h1>',
                    unsafe_allow_html=True)
start = st.sidebar.date_input("Select your comparator date range.",[min(df['ProdnDate']), max(df['ProdnDate'])])
st.sidebar.markdown(f'<h5 style="color:#C32148; font-size:13px;">{"Default would be the month before queried range."}</h1>',
                    unsafe_allow_html=True)

if start:
    lastMonthlastday = start[1]
    lastMonthfirstday = start[0]
else:
    # DEFAULT previous month for comparison
    startDatefirstday = pd.to_datetime(startDate).replace(day=1)
    # st.write(startDatefirstday)
    lastMonthlastday = startDatefirstday - datetime.timedelta(days=1)
    lastMonthfirstday = lastMonthlastday.replace(day=1)
    lastMonthlastday = lastMonthlastday.date()
    lastMonthfirstday = lastMonthfirstday.date()

#Get the first day of the lastest month in the choose period
latestmonthfirstday = max(df['ProdnDate']).replace(day=1)
# st.write(latestmonthfirstday.strftime("%#d/%#m/%Y"))

page_function_mapping[lens](df, automation, startDate, endDate, lastMonthfirstday, lastMonthlastday, latestmonthfirstday)










# read csv from a Excel file
# @st.experimental_memo
# def get_data() -> pd.DataFrame:
#     return pd.read_excel(dataset_location, sheet_name= automation)

# df = get_data() 

# if connectDataButton == False:
#     df = pd.read_excel(dataset_location, sheet_name= automation)
    
# else:
#     df = new_data
#     # df = pd.read_excel(dataset_location, sheet_name= automation)

# df = pd.read_excel(dataset_location, sheet_name= automation)
# page_function_mapping[view](df, automation)
