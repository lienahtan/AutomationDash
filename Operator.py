import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.graph_objs import *
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px 
import hydralit_components as hc
import time
from utils import print


def operator(df, automation, startDate, endDate, lastMonthfirstday, lastMonthlastday, latestmonthfirstday):
   
    st.title("Operator Insights for " + automation)
    placeholder = st.empty()
    
    # -------------------------------------------------------  getting data -------------------------------------------------------
    # seting date to datetime object 
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)

    placeholder = st.empty()

    format = go.Layout(
        margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin(
                b=0, #bottom margin
                t=0, #top margin
            ),
        # plot_bgcolor="#FFFFFF"
        )
    
    # renewing data
    perioddata = df[(df['ProdnDate'] >= startDate) & (df['ProdnDate'] <= endDate)]
    noofdays = (endDate - startDate).days
    
    prevStartWindow = pd.to_datetime(lastMonthfirstday)
    prevEndtWindow = pd.to_datetime(lastMonthlastday)
    prevdata = df[(df['ProdnDate'] >= prevStartWindow) & (df['ProdnDate'] <= prevEndtWindow )] 
    
    # Grouping data by modules
    groupbydf = perioddata.copy()
    groupbydf = groupbydf.groupby(['Module'])
    
    data = perioddata.values.tolist()
    dates = []
    modules = []
    operators = []adfadad

    for row in data:
        if row[0].strftime("%#d/%#m/%Y") not in dates:
            dates.append(row[0].strftime("%#d/%#m/%Y"))
        if row[6] not in modules:
            modules.append(row[6])
        if row[7] not in operators:
            operators.append(row[7])
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    # -----------------------------------------------------------nagivation bar -------------------------------------------------------
    # specify the primary menu definition
    menu_data = [
            {'icon':"🐙",'label':"Individual Operators"},
            {'icon': "far fa-address-book", 'label':"View Dataset"},
            {'icon': "💀", 'label':"Guide"},
            {'icon': "far fa-chart-bar", 'label':"Empty"},#no tooltip message
    ]
    
    # we can override any part of the primary colors of the menu
    over_theme = {'txc_inactive': 'black','menu_background':'#ADD8E6','txc_active':'#0047AB','option_active':'white'}
    # over_theme = {'txc_inactive': '#FFFFFF'} 
    
    with placeholder.container():
        menu_id = hc.nav_bar(menu_definition=menu_data, home_name='Overview', override_theme=over_theme, sticky_mode='pinned')
        
        if menu_id == "Overview":
            
            st.subheader("For period " + str(startDate.date()) + ' to ' + str(endDate.date()))
            
            main1, main2, main3, main4   = st.columns((0.5,0.5,1,1))
            
            with main1:
                operatorNames = ''
                for i in operators:
                    operatorNames += i
                    operatorNames += '\n\n'
                st.markdown(f'<h2 style="text-align: left; color:#2F847C; font-size:18px;">{"Operators present:"}</h1>', unsafe_allow_html=True)
                st.success(operatorNames)
            
            with main2:
                st.markdown(f'<h2 style="text-align: left; color:black; font-size:18px;">{"Best Operator: "}</h1>', unsafe_allow_html=True)
                st.info('Alvin')
                st.markdown(f'<h2 style="text-align: left; color:black; font-size:18px;">{"Most Improved Operator: "}</h1>', unsafe_allow_html=True)
                st.info('Nathaniel')
                
            with main3:
                hc.info_card(title='Average Operators EXP', content='All good!', sentiment='good', bar_value=77, icon_size="2.4rem",title_text_size="1.8rem",content_text_size="1.4rem")
                with hc.HyLoader('Loading',hc.Loaders.pulse_bars,):
                    time.sleep(0.5)
                    
            with main4:
                hc.info_card(title='Empty Field', content='Empty Field', sentiment='good', bar_value=77, icon_size="2.4rem",title_text_size="1.8rem",content_text_size="1.4rem")
                with hc.HyLoader('Loading',hc.Loaders.pulse_bars,):
                    time.sleep(0.5)
                    
                    
            info1, info2, info3   = st.columns((0.5,1,1))
            
            with info1:
                st.subheader("Per Alarm Type")
                
            with info2:
                st.subheader("Most Recovery Done 1EXP")
                
            with info3:
                st.subheader("Fastest Reovery Time 1EXP")
                
                
                
            info11, info22, info33   = st.columns((0.5,1,1))
            
            with info1:
                st.subheader("Cumulative")
                
            with info2:
                st.subheader("Most experience based on count 1EXP")
                
            with info3:
                st.subheader("Most Experience based on time 1EXP")
            
            
            
            
            
            
        if menu_id == "Individual Operators":
            
            operator1, operator2 = st.columns((1,1))
            
            with operator1:
                st.header('Operator')

            with operator2:
                st.header('EXP Chart')
    
        
        if menu_id == "View Dataset":
            st.write(perioddata)
        
        
        
        
        
                    
