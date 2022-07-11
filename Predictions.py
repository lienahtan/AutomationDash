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


def predicting(perioddata, dateDict, component_selected, rootcause_selected,):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
    filteredDf = filteredDf.loc[perioddata['Diagonstics'] == rootcause_selected]
    
    timeDict = {}
    
    timeDict.update({rootcause_selected: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[row[8]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
    preddf = pd.DataFrame(timeDict.values.tolist())
    
    st.write(preddf)
    
