from cProfile import label
import datetime
from datetime import timedelta
from itertools import count, groupby
from optparse import TitledHelpFormatter
from re import sub
from sys import modules
from tracemalloc import start
from turtle import width
from unicodedata import name
from numpy import average, size
import pandas as pd
from pyparsing import null_debug_action
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



# calculating outliers
# timedict takes in a dict with key being the modulue, value has {date: value}
def findoutliers(timeDict):
    outputstr = ''
    output = []
    for key,value in timeDict.items():
        zscores = stats.zscore(list(value.values()))
        noofmorethan3 = len([i for i in zscores if i >= 3])
        if noofmorethan3 > 0:
            # getting all outliers
            timeagainstoutliers = [a*b for a,b in zip(list(value.values()),
                                              [1 if i>=3 else 0 for i in zscores])]
            output.append([key, 
                           [i for i in timeagainstoutliers if i >= 3]])
            
    for name,outlier in output:
        outputstr += name + ' - ' + str(outlier)
        outputstr += '\n'
            
    return outputstr