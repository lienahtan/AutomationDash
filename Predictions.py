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


def predicting(perioddata, dateDict, filter, format):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == filter]
    
    timeDict = {}
    
    timeDict.update({filter: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[row[5]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
    timeDist = go.Figure(layout = format)
    value = timeDict[filter]