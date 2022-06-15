import datetime
import altair as alt
import pandas as pd
from pyparsing import null_debug_action
import streamlit as st
import pdfkit
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from datetime import date
import streamlit as st
from streamlit.components.v1 import iframe
from streamlit_echarts import st_echarts



def module(data: pd.DataFrame, lens, startDate, endDate):
   
    st.title("Module OVERVIEW for " + lens)
    # options = {
    #     "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
    #     "series": [
    #         {
    #             "name": "Pressure",
    #             "type": "gauge",
    #             "axisLine": {
    #                 "lineStyle": {
    #                     "width": 10,
    #                 },
    #             },
    #             "progress": {"show": "true", "width": 10},
    #             "detail": {"valueAnimation": "true", "formatter": "{value}"},
    #             "data": [{"value": [1,2,3,4], "name": "Score"}],
    #         }
    #     ],
    # }
    
    # st.write('hello')
    # st_echarts(options=options, width="100%", key=0)
    

        
    
        
    
        