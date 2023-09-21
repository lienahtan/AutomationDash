
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.graph_objs import *
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px 
from utils import hasitimproved, moduledistributionChart, alarmsTable, componenttimeChart, componentcountChart, componentdistributionChart
from reco import findoutliers
from utils import automationAvail, counttimeDate, componentmeantime, faulttimeChart, faultcountChart
from utils import modulecountChart, moduletimeChart, paretoChart, componentTable, faultdistributionChart
from Predictions import predicting
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report
# from pandas_profiling import ProfileReport
import calendar
from streamlit_echarts import st_echarts
from df_creation import get_dateDict
import hydralit_components as hc
from MLPred import MLPred


def overall(df, automation, startDate, endDate, lastMonthfirstday, lastMonthlastday, latestmonthfirstday):
    
    # seting date to datetime object 
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)
    
    st.title("Automation Overview for " + automation)
    st.write("*ONLY FOR DEMO PURPOSES* - Best viewed in Light mode")

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
    perioddata = df[((df['ProdnDate'] >= startDate) & (df['ProdnDate'] <= endDate))]
    noofdays = (endDate - startDate).days
    
    # getting previous sliding date window
    # prevStartWindow = startDate + timedelta(days=-noofdays)
    # prevEndtWindow = startDate + timedelta(days=-1)
    # prevStartWindow = pd.to_datetime(prevStartWindow)
    # prevEndtWindow = pd.to_datetime(prevEndtWindow)
    prevStartWindow = pd.to_datetime(lastMonthfirstday)
    prevEndtWindow = pd.to_datetime(lastMonthlastday)
    prevdata = df[((df['ProdnDate'] >= prevStartWindow) & (df['ProdnDate'] <= prevEndtWindow ))] 
    

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
        
        menu_data = [
            {'icon': "far fa-address-book", 'label':"View Dataset"},
            {'icon': "far fa-chart-bar", 'label':"Pandas Profiling"},
        ]
        
        # we can override any part of the primary colors of the menu
        over_theme = {'txc_inactive': 'black','menu_background':'#ADD8E6','txc_active':'#0047AB','option_active':'white'}
        # over_theme = {'txc_inactive': '#FFFFFF'} 
        menu_id = hc.nav_bar(menu_definition=menu_data, home_name='Dashboard', override_theme=over_theme)
        
        if menu_id == "Pandas Profiling":            
            pass
            # profile = ProfileReport(df, title='Machine data',
            #                     variables = {
            #                         "descriptions": {
            #                         'ProdnDate' : '1',
            #                         'ProdnShift' : '2',
            #                         'StartTime' : '3',
            #                         'EndTime' : '4',
            #                         'Duration(mins)' : '5',
            #                         'DT Reason Detail' : '6',
            #                         'Module' : '7'
            #                         }
            #                     }
            #                     )
#             st.dataframe(data=st_profile_report(profile))
            st.write("Report loaded. Please approach admin for dataset.")

        
        if menu_id == "View Dataset":
            st.write("Please approach admin for dataset.")
#             st.write(perioddata)
            
            
        if menu_id == "Dashboard":
            # --------------------Summary -------------------

            guage_1, key_indicators ,key_metrics, key_metrics2 = st.columns((3.5,2.5,4, 4))
                
        
            # SUMMARY AND RECOMMENDATIONS
            with key_metrics:
                st.subheader('Key Figures')
                
            #HOW TO PUT COLOURED WORds
            #     st.markdown(f'<h1 style="text-align: center; color:black; font-size:20px;">{"Total Downtime in minutes⏳"}</h1>', unsafe_allow_html=True)

            # fill in those three columns with respective metrics or KPIs
            dt = perioddata['Duration(mins)'].sum()
            prevdt = prevdata['Duration(mins)'].sum()
            
            # Auto availability
            # calculated availability
            availability = round((((noofdays * 8.5 * 60) - dt)/ (noofdays * 8.5 * 60)) , 3)*100
        #    colours for gauge
            #          'steps': [{'range':[0,0.1], 'color': '#FF0000'},
            #                    {'range':[0.1,0.2], 'color': '#FF3000'},
            #                    {'range':[0.2,0.3], 'color': '#FF6000'},
            #                    {'range':[0.3,0.4], 'color': '#FF9000'},
            #                    {'range':[0.4,0.5], 'color': '#FFC000'},
            #                    {'range':[0.5,0.6], 'color': '#FFF000'},
            #                    {'range':[0.6,0.7], 'color': '#CCFF00'},
            #                    {'range':[0.7,0.8], 'color': '#99FF00'},
            #                    {'range':[0.8,0.9], 'color': '#55FF00'},
            #                    {'range':[0.9, 1], 'color': '#00FF00'}],
            options = {
        "series": [
            {
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "radius": '90%',
                "min": 0,
                "max": 100,
                "center": ["40%", "60%"],
                "splitNumber": 10,
                "axisLine": {
                    "lineStyle": {
                        "width": 15,
                        "color": [
                            [0.25, "#FF403F"],
                            [0.5, "#ffa500"],
                            [0.75, "#FDDD60"],
                            [1, "#64C88A"],
                        ],
                    }
                },
                "pointer": {
                    "icon": "path://M12.8,0.7l12,40.1H0.7L12.8,0.7z",
                    "length": "42%",
                    "width": 13,
                    "offsetCenter": [0, "-40%"],
                    "itemStyle": {"color": "black"},
                },
                "axisTick": {"length": 20, "lineStyle": {"color": "auto", "width": 1}},
                "splitLine": {"length": 35, "lineStyle": {"color": "auto", "width": 5}},
                "axisLabel": {
                    "color": "#464646",
                    "fontSize": 12,
                    "distance": -60,
                },
                "title": {"offsetCenter": [0, "-20%"], "fontSize": 20},
                "detail": {
                    "fontSize": 30,
                    "offsetCenter": [0, "0%"],
                    "valueAnimation": True,
                    "color": "auto",
                    "formatter": "{value}%",
                },
                "data": [{"value": round(availability,2) , "name": ""}],
            }
        ]
    }

            with guage_1:
                st.subheader('For ' +  str(startDate.date()) + ' to ' + str(endDate.date()))
                st.markdown("Vs " + str(lastMonthfirstday) + ' - ' + str(lastMonthlastday))
                st_echarts(options, width="450px", height="350px", key="gauge")
              
            
            # Adding to the 3 main metrics
            fig1 = go.Figure(layout = format)
            
            fig1.add_trace(go.Indicator(
                mode = "number+delta",
                value = dt,
                title = {"text": "<br><span style='font-size:20;color:Black'>Total Downtime (mins) ⏳</span>"},
                delta = {'reference': prevdt, 'relative': True, 'increasing': {'color': 'red'},
                                    'decreasing': {'color': 'green'}, 'valueformat':'.2%'},
                number={"font":{"size":50}},
                domain = {'x': [0, 1], 'y': [0, 0.7]}))
            fig1.update_layout(autosize=True,
                width=50,
                height=130,
                paper_bgcolor= "#ADD8E6")
            with key_indicators:
                st.plotly_chart(fig1, use_container_width= True)
            
            count = perioddata.shape[0]
            prevcountdt = prevdata.shape[0]
            fig2 = go.Figure(layout = format)
            
            fig2.add_trace(go.Indicator(
                mode = "number+delta",
                value = count,
                title = {"text": "<br><span style='font-size:20;color:Black'>Alarm Frequency 🗠</span>"},
                delta = {'reference': prevcountdt, 'relative': True, 'increasing': {'color': 'red'},
                                    'decreasing': {'color': 'green'}, 'valueformat':'.2%'},
                number={"font":{"size":50}},
                domain = {'x': [0, 1], 'y': [0, 0.7]}))
            fig2.update_layout(autosize=True,
                width=50,
                height=130,
                paper_bgcolor= "#ADD8E6")
            with key_indicators:
                st.plotly_chart(fig2, use_container_width= True)
                
                
            fig3 = go.Figure(layout = format)
            
            fig3.add_trace(go.Indicator(
                mode = "number+delta",
                value = dt/count,
                title = {"text": "<br><span style='font-size:20;color:Black'>Average Downtime ➗</span>"},
                delta = {'reference': prevdt/prevcountdt, 'relative': True, 'increasing': {'color': 'red'},
                                    'decreasing': {'color': 'green'}, 'valueformat':'.2%'},
                number={"font":{"size":50}},
                domain = {'x': [0, 1], 'y': [0, 0.7]}))
            fig3.update_layout(autosize=True,
                width=50,
                height=130,
                paper_bgcolor="#ADD8E6")
            with key_indicators:
                st.plotly_chart(fig3,use_container_width= True)
            
        
            # AA tracking across time
            rowauto_spacer1, rowauto_1, rowauto_spacer2 = st.columns((.2, 7.1, .2))
            
            with rowauto_1:
                st.subheader('a) Automation Availability (AA)')
                st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:18px;">{"Investigate %change in AA"}</h1>', unsafe_allow_html=True)        
        
            customnoofdays =  (max(df['ProdnDate']) - min(df['ProdnDate'])).days
            customdateDict = get_dateDict(min(df['ProdnDate']), customnoofdays)
            
            # dateandtime gives TOTAL DOWNTIME for each day
            dateandtime = automationAvail(df, customdateDict)
            dateandtime = pd.DataFrame.from_dict(list(dateandtime.items()))       
            # interval = st.radio("Select time interval", ('Week', 'Month'), horizontal = True)
            
            dateandtime = dateandtime.rename({0: 'Date', 1: 'Downtime'}, axis=1)
            dateandtime['Date'] = pd.to_datetime(dateandtime['Date'], format="%d/%m/%Y")
            dateandtime['Year'] = pd.DatetimeIndex(dateandtime['Date']).year
            dateandtime['Month'] = pd.DatetimeIndex(dateandtime['Date']).month
            dateandtime['Month'] = dateandtime['Month'].apply(lambda x: calendar.month_abbr[x])
            dateandtime['Date'] =  dateandtime['Month'] + dateandtime['Year'].apply(lambda x : str(x))
            dateandtime.drop("Year", axis=1, inplace=True)
            dateandtime.drop("Month", axis=1, inplace=True)
        

            # dateandtime now consist of eg. Feb20222 | 1500(mins)
            # finding out how many days per month to calculate total AA
            noofdayspermonth = {}
            monthyear = dateandtime['Date']
            for unique in monthyear:
                noofdayspermonth[unique] = noofdayspermonth.get(unique, 0) + 60*8.5
            
            dateandtime = dateandtime.groupby(["Date"], sort = False).sum()
            # st.write(dateandtime)
                    
            autoavail = dateandtime
            autoavail['AA'] = autoavail['Downtime'].div(list(noofdayspermonth.values()))
            autoavail['AA'] = autoavail['AA'].div(-1)
            autoavail = 1 + autoavail
            # st.write(autoavail)
            autoavail['perchange'] = autoavail['AA'].pct_change()
            autoavail['perchange'] = autoavail['perchange'] * 100
            
            
            fig = make_subplots(specs=[[{"secondary_y": True}]], vertical_spacing = 0.0)
            fig.add_trace(go.Bar(name='AA',y=autoavail['AA'], x = autoavail.index.tolist(), text = round(autoavail['AA'], 3),
                                textposition=['inside']*len(noofdayspermonth),), secondary_y = False)
            
            fig.add_trace(go.Scatter(name= 'Change in AA', y=autoavail['perchange'], x = autoavail.index.tolist(), text = round(autoavail['perchange'],2),
                                    textposition="middle right", mode="lines+text+markers",
                                    marker=dict(color='#1B1811', size=5),textfont=dict(color='#C41E3A', size= 28)), secondary_y= True)
            fig.update_layout(
                autosize=True,
                width=1600,
                height=350,
                bargap=0.60,
                margin=dict(l=0,r=0,b=0,t=0),
                uniformtext_minsize=25, uniformtext_mode='hide',
                paper_bgcolor="#FFFFFF"
            )
            
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.5
                ))
            
            st.plotly_chart(fig, use_container_width= True)    
        
        
            # # -------------------------Pareto Chart--------------------------------
            
            
            row2_spacer1, row2_1, row2_spacer2 = st.columns((.2, 7.1, .2))
            
            with row2_1:
                st.subheader('b) Detailed breakdown of modules affecting AA')
                st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:18px;">{"Investigate which modules are causing the most downtime/ frequency."}</h1>', unsafe_allow_html=True)        
            tableG1, tableG2 = st.columns((1,1))
            
            with tableG1:
                tableORgraph1 = st.checkbox('Show Total Downtime Table')
            
            with tableG2:
                tableORgraph2 = st.checkbox('Show Alarm Frequency Table')

            row2_col1, row2_col2 = st.columns((1,1))            
            
            with row2_col1:
                timechart = paretoChart(perioddata, 'time')
                
                # paretoTimeOutput = '🔹 ' + str(chart[1])
                causesReco1 = timechart[1]
                
                if tableORgraph1:
                    st.table(timechart[2])
                else:
                    st.plotly_chart(timechart[0], use_container_width=True)
                
            with row2_col2:
                # chart's second output is a top cause of failure
                chart = paretoChart(perioddata, 'counts')
                
                if autoavail['perchange'][-1] < 0:
                    sign = 'down '
                else:
                    sign = 'up '
                causesReco2 = chart[1]
                
                AAchangedate = str(endDate.month_name()[:3]) + str(endDate.year) 
                
                Reco1change = round(autoavail['perchange'][AAchangedate],2)
                
                chartReco1 = 'AA is ' + sign + str(Reco1change) + '%'
         
                                    
                if tableORgraph2:
                    st.table(chart[2])
                else:
                    st.plotly_chart(chart[0], use_container_width=True)
        
        
            # General Charts row 1 Distribution of Alarms Timings Over Time-------------------
            row1_spacer1, row1_1, row1_spacer2 = st.columns((.2, 7.1, .2))
            with row1_1:
                st.subheader('c) Analysis per Module')
                st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:18px;">{"Investigate the time trend of (Total time/ Count) of a module."}</h1>', unsafe_allow_html=True)
                
            row1_col1, row1_col2, row1_col4  = st.columns((.1, 2.4, 4.6))
            
            with row1_col2:   
                queryOptions = timechart[2].index.tolist()
                module_selected = st.selectbox ("Which module do you want to analyze?", queryOptions)
                measure_selected = st.selectbox ("Which measure do you want to analyze?", ['Total time', 'Count', 'Distribution'])
#                 st.markdown('Components of selected module (Scroll)')
                componentBreakdown = componentTable(perioddata, format, module_selected, prevdata)
#                 st.write(componentBreakdown[0], use_container_width=True)
                
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
                        chart = moduledistributionChart(perioddata, dateDict, module_selected)
                        st.plotly_chart(chart,use_container_width=True)
                
                    
        # ----------------------RECOMENDATION improvement over the last 30 days!!------------------ "'s performance has become much worse over the last 30days!"
            # reco for sudden change in improvments or lags
            improvements = hasitimproved(alarmDict)
            output = ''
            for i in improvements[0]:
                output += i
                
            for j in improvements[1]:
                output += j
                
            chartreco2 = output
            

            row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
            with row3_1:
                st.subheader("d) Further Analysis of Module's Alarms")
                st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:18px;">{"Investigate the time trend of (Total time/ Count) per component (as per module selected above)."}</h1>', unsafe_allow_html=True)

                            
            row3_col1, row3_col2, row3_col4  = st.columns((.1, 2.4, 4.6))
                
            with row3_col2:
            
                filteredDf = perioddata.loc[perioddata['Module'] == module_selected]
                groupbydf = filteredDf.copy()
                groupbydf = groupbydf.groupby(['DT Reason Detail'])
            
                # to get keys for the groupby, replaced by getting it directly from the dataframe
                uniqueComponents = list(groupbydf.groups.keys())
                # also known as alarm selected
                alarmquery = componentBreakdown[1]['Component'].tolist()
                component_selected = st.selectbox ("Which alarm do you want to analyze?", alarmquery)
                component_measure_selected = st.selectbox ("Select measure you want to analyze", ['Total time', 'Count', 'Distribution'])
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
                

                        
            # ----------------------Filter for root cause of the issue!!----------------- 
            row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
            with row3_1:
                st.subheader("e) Root Causes of Alarm")
                st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:18px;">{"Finding the root cause for alarm selected above."}</h1>', unsafe_allow_html=True)

            row3_col1, row3_col2, row3_col4  = st.columns((.1, 2.4, 4.6))
                
            with row3_col2:
                filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
                groupbydf = filteredDf.copy()
                groupbydf = groupbydf.groupby(['Diagonstics'])
            
                uniqueAlarms = list(groupbydf.groups.keys())
                
                rootcause_selected = st.selectbox ("Which root cause do you want to analyze?", uniqueAlarms)
                rootcause_measure_selected = st.selectbox ("Select measure.", ['Total time', 'Count', 'Distribution'])
                st.markdown('Root causes of selected alarm (Scroll)')
                st.plotly_chart(alarmsTable(perioddata, format,component_selected, rootcause_selected, prevdata), use_container_width=True)
                st.markdown('Average Fault Frequency')
                componentmeantime(perioddata, dateDict, component_selected, rootcause_selected)
                
            with row3_col4:
                
                if rootcause_measure_selected == 'Total time':
                    st.plotly_chart(faulttimeChart(perioddata, dateDict, component_selected, rootcause_selected, format),
                                use_container_width=True)
                elif rootcause_measure_selected == 'Count':
                    st.plotly_chart(faultcountChart(perioddata, dateDict, component_selected, rootcause_selected, format),
                                use_container_width=True)
                elif rootcause_measure_selected == 'Distribution':
                    st.plotly_chart(faultdistributionChart(perioddata, dateDict, component_selected, rootcause_selected),
                                use_container_width=True)
          

            # ----------------------RECOMENDATION OUTLIERS---------------------
            outliers = findoutliers(timeDict)
                        
                        
            # ----------------------RECOMENDATION CORRELATION---------------------            
            timeDF = pd.DataFrame(timeDict).corr().unstack().sort_values(kind='quicksort')
            timeDF = timeDF[abs(timeDF) >= 0.5]
            namesforcorr = timeDF.index.tolist()
            corrmorethan05 = timeDF.values.tolist()
            corroutput = ''
            
            for i in range(len(corrmorethan05)):
                if abs(corrmorethan05[i]) != 1:
                    corroutput += " ⚠ " + str(namesforcorr[i]) + ' - ' + str(round(corrmorethan05[i],3))
                    corroutput += "\n\n"
                
            # investigation if time of day affects count and downtime
            rowe_spacer1, rowe_1, rowe_spacer2 = st.columns((.2, 7.1, .2))
            with rowe_1:
                st.subheader("f) Analysis of Alarm Frequency and Downtime with Time of Day")
                st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:18px;">{"Investigate if time of day affects count and downtime."}</h1>', unsafe_allow_html=True)
                            
            rowe2_col1, rowe2_col3  = st.columns((.1, 4.6))
            
            with rowe2_col3:
                component_selected = st.selectbox ("Select component to view", alarmquery)
                st.plotly_chart(counttimeDate(perioddata, component_selected), use_container_width= True)
                
           
            # -----------------------------------Detailed outliers-----------------------------
        

                    
                
            # --------------------------------------ALL RECOMMENDATIONS ---------------------------------------------------
            with key_metrics:
                chartReco = st.expander("〽️ Over the last 30 days:", expanded=True)
                with chartReco:
                    st.warning(chartReco1)
                    st.warning(chartreco2)
                
                if outliers == 'No outliers':
                    with key_metrics:
                        st.success('✅ Good! No outliers found')
                    
                else:
                    see_outliers = st.expander('🖱️ Possible outlliers found.')
                    with see_outliers:
                        st.error(outliers)
                        
                if corroutput == '':
                    st.info('✅ Good! No correlation between modules found.')
                else:
                    with key_metrics:
                        see_corr = st.expander('⚠ Significant Module Correlation Found')
                        with see_corr:
                            st.error(corroutput)
                    
                    
            with key_metrics2:
                downtimecausesReco = st.expander("➿ Top 3 Module Downtime: ", expanded=True)
                with downtimecausesReco:
                    st.info(causesReco1)
                   
                    
                freqcausesReco = st.expander("➿ Top 3 Module Count: ", expanded=True)
                with freqcausesReco:
                    st.info(causesReco2)
            # ---------------------prediction for component level---------------------
            row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
            with row4_1:
                st.subheader("g) Forecasting using Machine Learning")
                st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:18px;">{"Forecasts the downtime of an alarm."}</h1>', unsafe_allow_html=True)
            
            
            row4_col1, row4_col2  = st.columns((0.1, 4.6))

            filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
            groupbydf = filteredDf.copy()
            groupbydf = groupbydf.groupby(['Diagonstics'])

            with row4_col2:
                if noofdays < 60:
                    st.write('Not enough days for prediction.')
                else:
                    # st.plotly_chart(faulttimeChart(perioddata, dateDict, component_selected, rootcause_selected, format),
                    #                 use_container_width=True)
                    # st.write(predicting(perioddata, dateDict, component_selected, rootcause_selected, endDate, period))
                    # st.write(MLPred(perioddata, dateDict, component_selected, rootcause_selected, endDate, period))
                    alarmquery = perioddata['DT Reason Detail'].unique().tolist()
                    st.write(MLPred(perioddata, dateDict, alarmquery, rootcause_selected, endDate))


            # st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
                        
            
            
                
            
                    
