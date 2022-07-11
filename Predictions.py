import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import OrderedDict
import chart_studio.plotly as py
from plotly.graph_objs import *
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px 
import plotly.figure_factory as ff
import numpy as np
import scipy.stats as stats
import statistics
import datetime


def predicting(perioddata, dateDict, component_selected, rootcause_selected, endDate, period):
    period = 7
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
    filteredDf = filteredDf.loc[perioddata['Diagonstics'] == rootcause_selected]
    
    timeDict = {}
    
    timeDict.update({rootcause_selected: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[row[8]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
          
#     preddf = pd.DataFrame.from_dict(timeDict.values()).T
#     preddf = preddf.rename(columns={preddf.columns[0]: 'Date'})
    
   
    date = list(list(timeDict.values())[0].keys())
    count = list(list(timeDict.values())[0].values())
    d = {'y':date,'ds':count}
    preddf = pd.DataFrame(d)
#     preddf.columns = ['y', 'ds']
    st.write(preddf)
    
    datelist = pd.date_range(endDate + datetime.timedelta(days=1) , periods=period + 1)
    indexDate = []
    for date in datelist:
        indexDate.append(date.strftime("%#d/%#m/%Y"))
    st.write(indexDate)
    
        
    
