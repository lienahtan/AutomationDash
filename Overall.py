from datetime import timedelta
from enum import auto
from lib2to3.pgen2.pgen import DFAState
from this import d
from tokenize import group
from tracemalloc import start
from numpy import average, size
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import chart_studio.plotly as py
from plotly.graph_objs import *
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px 
from utils import distributionChart, moduleTable, componenttimeChart, componentcountChart, componentdistributionChart
from reco import findoutliers
from utils import automationAvail, local_css
from utils import modulecountChart, findvalueHolder, moduletimeChart, paretoChart, componentTable
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport


def overall(df: pd.DataFrame, lens, startDate, endDate):
    
    st.title("AUTOMATION OVERVIEW for " + lens)

    placeholder = st.empty()
    
    
    global reco1
    reco1 = ''
            
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
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)
    perioddata = df[(((df['ProdnDate'] >= startDate) & (df['ProdnDate'] <= endDate)) | 
                ((df['ProdnDate'] >= startDate) & (df['ProdnDate'] <= endDate)))]
    noofdays = (endDate - startDate).days
    
    # getting previous sliding date window
    prevStartWindow = startDate + timedelta(days=-noofdays)
    prevEndtWindow = startDate + timedelta(days=-1)
    prevStartWindow = pd.to_datetime(prevStartWindow)
    prevEndtWindow = pd.to_datetime(prevEndtWindow)
    prevdata = df[(((df['ProdnDate'] >= prevStartWindow) & (df['ProdnDate'] <= prevEndtWindow )) | 
                ((df['ProdnDate'] >= prevStartWindow) & (df['ProdnDate'] <= prevEndtWindow )))] 

    # Grouping data by modules
    groupbydf = perioddata.copy()
    groupbydf = groupbydf.groupby(['Module'])
    
    
    with placeholder.container():
        # ----------------formatting data-------------------
        # creating unique dates and MODULE from data
        
        
        data = perioddata.values.tolist()
        dates = []
        modules = []

        for row in data:
            if row[0].strftime("%#d/%#m/%Y") not in dates:
                dates.append(row[0].strftime("%#d/%#m/%Y"))
            if row[6] not in modules:
                modules.append(row[6])
            

        # creating range of all dates from data

        datelist = pd.date_range(startDate, periods=noofdays + 1)

        indexDate = []
        for date in datelist:
            indexDate.append(date.strftime("%#d/%#m/%Y"))
        
        
        dateDict = {}
        timeDict = {}
        alarmDict = {}


        for date in indexDate:
            dateDict.update({date: 0})

        #creating correlation matrix for COUNT and DURATION

        for module in modules:
            timeDict.update({module: dateDict.copy()})
            alarmDict.update({module: dateDict.copy()})
            
            
        for row in data:
            timeDict[row[6]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
            alarmDict[row[6]][row[0].strftime("%#d/%#m/%Y")] += 1
        
        
        # ----------------visualisation-------------------        
        
        row0_spacer1, row0_1, row0_spacer2 = st.columns((.2, 7.1, .2))
        with row0_1:
            st.markdown("")
            see_data = st.expander('You can click here to see the raw data first 👉👉👉')
            with see_data:
                st.write(perioddata)
        
        # # create space
        # st.markdown('##')
        
        guage_spacer1, guage_1 = st.columns((2.5, 5))
            
       
        # SUMMARY AND RECOMMENDATIONS
        with guage_1:
            st.subheader('Summary')
            load_data = st.checkbox('Click here to view dataset detailed summary 👇 at the bottom of the page')
            st.markdown('Recommendations')
            st.markdown('🔹 Avaiablility is down 18% from last week')
            
            
        kpiword1, kpiword2, kpiword3 = st.columns(3)
        with kpiword1:
            st.markdown(f'<h1 style="text-align: center; color:black; font-size:20px;">{"Total Downtime in minutes⏳"}</h1>', unsafe_allow_html=True)
        with kpiword2:
            st.markdown(f'<h1 style="text-align: center; color:black; font-size:20px;">{"Downtime Count"}</h1>', unsafe_allow_html=True)
        with kpiword3:
            st.markdown(f'<h1 style="text-align: center; color:black; font-size:20px;">{"Empty field"}</h1>', unsafe_allow_html=True)

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)
        # fill in those three columns with respective metrics or KPIs
        dt = perioddata['Duration(mins)'].sum()
        prevdt = prevdata['Duration(mins)'].sum()
        
        # Auto availability
        # calculated availability
        availability = round((((noofdays * 8.5 * 60) - dt)/ (noofdays * 8.5 * 60)) , 3)
        # Create availability chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = availability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Automation Availability"},
            gauge = {'shape':'angular', 
                 'steps': [{'range':[0,0.1], 'color': '#FF0000'},
                           {'range':[0.1,0.2], 'color': '#FF3000'},
                           {'range':[0.2,0.3], 'color': '#FF6000'},
                           {'range':[0.3,0.4], 'color': '#FF9000'},
                           {'range':[0.4,0.5], 'color': '#FFC000'},
                           {'range':[0.5,0.6], 'color': '#FFF000'},
                           {'range':[0.6,0.7], 'color': '#CCFF00'},
                           {'range':[0.7,0.8], 'color': '#99FF00'},
                           {'range':[0.8,0.9], 'color': '#55FF00'},
                           {'range':[0.9, 1], 'color': '#00FF00'}],
                 'bar' : {'color' :'black', 'thickness' : 0.5},
                 
            'bar' : {'color' :"#ADD8E6", 'thickness' : 0.4},
            'axis' : {'range':[None, 1], 'tickformat' : '.0'}
            }), layout= format)
        
        fig.update_layout(autosize=True,
        width=400,
        height=350)

        with guage_spacer1:
            st.plotly_chart(fig, use_container_width=True)
        
        fig1 = go.Figure(layout = format)
        
        fig1.add_trace(go.Indicator(
            mode = "number+delta",
            value = dt,
            title = {"text": "<br><span style='font-size:2em;color:Black'>Total Downtime in minutes⏳</span>"},
            delta = {'reference': prevdt, 'relative': True, 'increasing': {'color': 'red'},
                                  'decreasing': {'color': 'green'}},
            number={"font":{"size":50}},
            domain = {'x': [0, 1], 'y': [0, 1]}))
        fig1.update_layout(autosize=True,
            width=50,
            height=100,
            paper_bgcolor= "#ADD8E6")
        with kpi1:
            st.plotly_chart(fig1, use_container_width= True)
        
        count = perioddata.shape[0]
        countdt = prevdata.shape[0]
        fig2 = go.Figure(layout = format)
        
        fig2.add_trace(go.Indicator(
            mode = "number+delta",
            value = count,
            title = {"text": "Total Downtime in minutes⏳<br><span style='font-size:0.em;color:gray'>Subsubtitle</span>"},
            delta = {'reference': countdt, 'relative': True, 'increasing': {'color': 'red'},
                                  'decreasing': {'color': 'green'}},
            number={"font":{"size":50}},
            domain = {'x': [0, 1], 'y': [0, 1]}))
        fig2.update_layout(autosize=True,
            width=50,
            height=100,
            paper_bgcolor= "#ADD8E6")
        with kpi2:
            st.plotly_chart(fig2,use_container_width= True)
            
            
        fig3 = go.Figure(layout = format)
        
        fig3.add_trace(go.Indicator(
            mode = "number+delta",
            value = count,
            title = {"text": "Total Downtime in minutes⏳<br><span style='font-size:0.em;color:gray'>Subsubtitle</span>"},
            delta = {'reference': countdt, 'relative': True, 'increasing': {'color': 'red'},
                                  'decreasing': {'color': 'green'}},
            number={"font":{"size":50}},
            domain = {'x': [0, 1], 'y': [0, 1]}))
        fig3.update_layout(autosize=True,
            width=50,
            height=100,
            paper_bgcolor="#ADD8E6")
        with kpi3:
            st.plotly_chart(fig3,use_container_width= True)
        
        
        
        # count = perioddata.shape[0]
        # countdt = prevdata.shape[0]
        # kpi2.metric(label="Downtime Count 💍", value= int(count)
        #             ,delta= float(count - countdt)
        #             ,delta_color = "inverse")
        
        # kpi3.metric(label="Automation Efficiency %", value= f"$ {round(3,2)} ", delta= - round(3/2) * 100) 

        st.caption("Change as of last period (eg. Choosen period 8/7 - 14/7 will be compared to 1/7 - 7/7)")
        st.markdown('##')
    
    
        rowauto_spacer1, rowauto_1, rowauto_spacer2 = st.columns((.2, 7.1, .2))
        
        with rowauto_1:
            st.subheader('Automation Availability for ' + str(startDate.date()) + ' to ' + str(endDate.date()))
    
        dateandtime = automationAvail(perioddata, dateDict)
        dateandtime = pd.DataFrame.from_dict(list(dateandtime.items()))

        interval = st.radio("Select time interval (Days)", ('7', '30'), horizontal = True)
        
        day = startDate.weekday()
        if day == 0:
            day = 'W-MON'
        elif day == 1:
            day = 'W-TUE'
        elif day == 2:
            day = 'W-WED'
        elif day == 3:
            day = 'W-THU'
        elif day == 4:
            day = 'W-FRI'
        elif day == 5:
            day = 'W-SAT'
        elif day == 6:
            day = 'W-SUN'
        
        if interval == '7':
            frequency = day
        elif interval == '30':
            frequency = 'M'
            
        #starts from StartDate and every interval after that
        # intervalDates = pd.date_range(startDate, periods= ((noofdays + 1) /int(interval)) + 1, freq = frequency)
        # st.write(intervalDates)
        # st.write(startDate + timedelta(7 - startDate.weekday()))
        # intervalHolder = []
        # for i in range(1 , len(intervalDates)):
        #     intervalHolder.append(60 * 8.5 * ((intervalDates[i] -  intervalDates[i-1]).days)) 
        # intervalHolder.append(60 * 8.5 * ((endDate -  intervalDates[len(intervalDates) - 1]).days))
        intervalDates = pd.date_range(startDate, periods= ((noofdays + 1) /int(interval)) + 1, freq = 'W-MON')
        dateandtime =  dateandtime.iloc[7 - startDate.weekday():]
       
        dateandtime = dateandtime.groupby(dateandtime.index // int(interval)).sum()
        
        intervalHolder = []
        for i in range(1 , len(intervalDates)):
            intervalHolder.append(60 * 8.5 * ((intervalDates[i] -  intervalDates[i-1]).days))
        intervalHolder.append(60 * 8.5 * ((endDate -  intervalDates[len(intervalDates) - 1]).days))
      
    
        autoavail = dateandtime.set_index(intervalDates)
        autoavail[1] = autoavail[1].div(intervalHolder)
        autoavail[1] = autoavail[1].div(-1)
        autoavail = 1 + autoavail
        autoavail['perchange'] = autoavail.pct_change()
        autoavail['perchange'] = autoavail['perchange'] * 100
        # st.write(intervalDates.tolist())
    

        fig = make_subplots(specs=[[{"secondary_y": True}]], vertical_spacing = 0.0)
        # fig = px.bar(y=autoavail[1], x = autoavail.index, title="Automation Availability Tracking " + "(Interval: " + interval + " days)")
        fig.add_trace(go.Bar(name = 'Automation Availability', y=autoavail[1], x = autoavail.index, text = round(autoavail[1], 2), textposition=['outside']*len(intervalDates)), secondary_y=False)
        fig.add_trace(go.Scatter(y=autoavail['perchange'], x = autoavail.index), secondary_y=True)
        fig.update_layout(
            title_text = "Automation Availability Tracking " + "(Interval: " + interval + " days)",
            autosize=True,
            width=1600,
            height=350,
            bargap=0.50,
            margin=dict(l=0,r=0,b=0,t=25),)
        st.plotly_chart(fig)    
        # st.write(autoavail)
    
        # # -------------------------Pareto Chart--------------------------------
        
        
        row2_spacer1, row2_1, row2_spacer2 = st.columns((.2, 7.1, .2))
        
        with row2_1:
            st.subheader('Pareto Charts')
            st.markdown('Investigate the percentage of each alarm time/count.') 
        
        tableG1, tableG2 = st.columns((1,1))
        
        with tableG1:
            tableORgraph1 = st.checkbox('Show Time table instead of graph')
        
        with tableG2:
            tableORgraph2 = st.checkbox('Show Count table instead of graph')

        row2_col1, row2_col2 = st.columns((1,1))            
        
        with row2_col1:
            chart = paretoChart(perioddata, 'time')
            
            paretoTimeOutput = '🔹 Pareto charts show that ' + str(chart[1]) + ' makes up 50\% of total downtime'
            
            if tableORgraph1 == True:
                st.table(chart[2])
            else:
                st.plotly_chart(chart[0], use_container_width=True)
            
        with row2_col2:
            chart = paretoChart(perioddata, 'counts')
            
            with guage_1:
                st.markdown(paretoTimeOutput + ' and ' + str(chart[1]) + ' makes up 50\% of total alarms.')
            
            if tableORgraph2 == True:
                st.table(chart[2])
            else:
                st.plotly_chart(chart[0], use_container_width=True)
    
    
        # General Charts row 1 Distribution of Alarms Timings Over Time-------------------
        row1_spacer1, row1_1, row1_spacer2 = st.columns((.2, 7.1, .2))
        with row1_1:
            st.subheader('Analysis per Module')
            
        row1_col1, row1_col2, row1_col4  = st.columns((.1, 2.4, 4.6))
        
        with row1_col2:
            st.markdown('Investigate the time trend of (Total time/ Count) of a module.')    
   
            module_selected = st.selectbox ("Which module do you want to analyze?", list(timeDict.keys()))
            measure_selected = st.selectbox ("Which measure do you want to analyze?", ['Total time', 'Count', 'Distribution'])
            st.markdown('Breakdown of selected module (Scroll)')
            st.plotly_chart(componentTable(perioddata, format, module_selected), use_container_width=True)
            
        with row1_col4:
            if module_selected == None:
                st.markdown(f'<h1 style="color:#FF0000; font-size:24px;">{"Please select a query date.”"}</h1>', unsafe_allow_html=True)
            else:
                if measure_selected == 'Total time':
                    chart = moduletimeChart(timeDict, format, module_selected)
                    st.plotly_chart(chart,use_container_width=True)
                elif measure_selected == 'Count':
                    chart = modulecountChart(alarmDict, format, module_selected)
                    st.plotly_chart(chart,use_container_width=True)
                elif measure_selected == 'Distribution':
                    chart = componentdistributionChart(alarmDict, format, module_selected)
                    st.plotly_chart(chart,use_container_width=True)
            
                



    row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
    with row3_1:
        st.subheader("Further Analysis per Component")
        
    row3_col1, row3_col2, row3_col4  = st.columns((.1, 2.4, 4.6))
        
    with row3_col2:
        st.markdown('Investigate the time trend of (Total time/ Count) per component (as per module selected above).') 
        filteredDf = perioddata.loc[perioddata['Module'] == module_selected]
        groupbydf = filteredDf.copy()
        groupbydf = groupbydf.groupby(['DT Reason Detail'])
    
        uniqueComponents = list(groupbydf.groups.keys())
        
        component_selected = st.selectbox ("Which component do you want to analyze?", uniqueComponents)
        component_measure_selected = st.selectbox ("Which measure dsdfdso you want to analyze?", ['Total time', 'Count', 'Distribution'])
        title = ['Total Time Spent per Alarm', 'Time (Minutes)', 'Count', 'Time per Count']
        
        
    with row3_col4:
        
        if component_measure_selected == 'Total time':
            st.plotly_chart(componenttimeChart(perioddata, dateDict, component_selected, format),
                        use_container_width=True)
        elif component_measure_selected == 'Count':
            st.plotly_chart(componentcountChart(perioddata, dateDict, component_selected, format),
                        use_container_width=True)
        elif component_measure_selected == 'Distribution':
            st.plotly_chart(componentdistributionChart(perioddata, dateDict, component_selected),
                        use_container_width=True)
        

                
    # st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
    
    # Detailed summary
    sum1_spacer1, sum1_1, sum1_2, sum1_spacer2 = st.columns((.2, 8, 2, 4))
    with sum1_1:
        if load_data == True:
            see_data = st.expander('Click to expand/ minimise detailed summary of data set')
            with see_data:
                profile = ProfileReport(df, title='Machine data',
                                    variables = {
                                        "descriptions": {
                                        'ProdnDate' : '1',
                                        'ProdnShift' : '2',
                                        'StartTime' : '3',
                                        'EndTime' : '4',
                                        'Duration(mins)' : '5',
                                        'DT Reason Detail' : '6',
                                        'Module' : '7'
                                        }
                                    }
                                    )
                st.dataframe(data=st_profile_report(profile))
                
    # ----------------------RECOMENDATION OUTLIERS---------------------
    outliers = findoutliers(timeDict)
    if outliers:
        with guage_1:
            # st.info('⚠ Warning! Outlliers Found')
            see_outliers = st.expander('⚠ Warning! Outlliers Found')
            with see_outliers:
                st.write(outliers)
    else:
        with guage_1:
            st.markdown('🔹 Good! No outliers found')
                
                
    # ----------------------RECOMENDATION CORRELATION---------------------            
    timeDF = pd.DataFrame(timeDict).corr().unstack().sort_values(kind='quicksort')
    timeDF = timeDF[abs(timeDF) >= 0.5]
    namesforcorr = timeDF.index.tolist()
    corrmorethan05 = timeDF.values.tolist()
    output = ''
    
    for i in range(len(corrmorethan05)):
        if abs(corrmorethan05[i]) != 1:
            output += " ⚠ " + str(namesforcorr[i]) + ' has high correlation of ' + str(corrmorethan05[i])
    if output == '':
        with guage_1:
            st.markdown('🔹 No correlation between modules found')
    else:
        with guage_1:
            see_corr = st.expander('⚠ Warning! High Correlation Found')
            with see_corr:
                st.write(output)
                
            
            
            
            
    
