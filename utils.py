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


def findvalueHolder(perioddata):
    
    groupbydf = perioddata.copy()
    groupbydf = groupbydf.groupby(['Module'])
    
    componentDict = []

    for key, item in groupbydf:
        totaltime = sum(item['Duration(mins)'])
        componentDict.append([key, totaltime, len(item)])
    
    
    # disttime_holder = {}
            
    # for key, value in timeDict.items():
    #     sumy = 0
    #     count = 0
    #     for key1, value1 in value.items():
    #         sumy += value1
    #         if value1 > 0:
    #             count += 1
    #     disttime_holder.update({key: [sumy, count]})
        
    # disttime_holder = OrderedDict(sorted(disttime_holder.items(), key=lambda item: item[1], reverse = True))
    
    keyHolder = []
    valueHolder = []
    countHolder = []
    for key,value,count in componentDict: 
        keyHolder.append(key)   
        valueHolder.append(value)
        countHolder.append(count)
        
        
    return [keyHolder, valueHolder, countHolder]
        


def moduleTable(timeDict, format, headers):
    disttime_holder = {}
            
    for key, value in timeDict.items():
        sumy = 0
        count = 0
        for key1, value1 in value.items():
            sumy += value1
            if value1 > 0:
                count += 1
        disttime_holder.update({key: [sumy, count]})
        
    disttime_holder = OrderedDict(sorted(disttime_holder.items(), key=lambda item: item[1], reverse = True))
    
        
    keyHolder = []
    valueHolder = []
    countHolder = []
    timepercountHolder = []
    for key,value in disttime_holder.items():    
        keyHolder.append(key)
        valueHolder.append(value[0])
        countHolder.append(value[1])
        timepercountHolder.append(round(value[0]/ value[1], -0))

    time_fig = go.Figure(data=[go.Table(
        columnwidth = [2,1,1,1],
        header=dict(values= headers,
                    fill_color= '#5DADEC',
                    align='left'),
        cells=dict(values=[keyHolder, valueHolder, countHolder, timepercountHolder],
                fill_color='lavender',
                align='left'))
    ])
    
    time_fig.update_layout(
        autosize=True,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        height=250,
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return time_fig


def moduletimeChart(timeDict, format, module_selected):
    timeDist = go.Figure(layout = format)
    value = timeDict[module_selected]
    rollingavg = value.copy()
    rollingavg = pd.DataFrame([rollingavg]).T
    
    rollingavg[ '7day_rolling_avg' ] = rollingavg[0].rolling(7).mean()
    
    # total = sum(value.values())
    # county = sum(i > 0 for i in list(value.values()))
    # averageTimeTaken = total/county
    # p = np.percentile(np.array(list(value.values())), 50)
            
    timeDist.add_trace(
            go.Scatter(
                # input all the names
                x = list(value.keys()),
                # input the data for that column
                y = list(value.values()),
                
                visible = True,
                mode="lines+text",
                text = ["" if v == 0 else v for v in list(value.values())],
                textposition="top center",
            )
        )
    
    timeDist.add_trace(
        go.Scatter(
            
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )          
        
    timeDist.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Total Downtime",
        width=750,
        height=500,
        showlegend = False
    )
    
    return timeDist


    
def modulecountChart(alarmDict, format, module_selected):
    timeDistCount = go.Figure(layout = format)
    value = alarmDict[module_selected]
    rollingavg = value.copy()
    rollingavg = pd.DataFrame([rollingavg]).T
    
    rollingavg[ '7day_rolling_avg' ] = rollingavg[0].rolling(7).mean()
    
    timeDistCount.add_trace(
            go.Scatter(
                # input all the names
                x = list(value.keys()),
                # input the data for that column
                y = list(value.values()),
                name = module_selected, 
                visible = True,
                mode="lines+text",
                text = ["" if v == 0 else v for v in list(value.values())],
                textposition="top center"
            )
        )
    
    timeDistCount.add_trace(
        go.Scatter(
            name = '7-day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg'],
            visible= True,
            mode = "lines"
        )
    ) 
    
    timeDistCount.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.25
                ))
        
    timeDistCount.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Alarm Frequency",
        width=750,
        height=500,
    )
    
    return timeDistCount

def distributionChart(timeDict, format, module_selected):
    # distPlot = go.Figure(layout= format)
    value = timeDict[module_selected]
    colors = ['rgb(0, 10, 100)']
    
    distPlot = ff.create_distplot([[i for i in list(value.values()) if i != 0]], [module_selected], colors=colors,
                            show_curve=True)
    
    # distPlot.add_trace(
    #             go.Histogram(
    #                 name = module_selected,
    #                 x = list(value.values()),
    #                 visible = True,
    #             )
    # )
    distPlot.update_layout(
        autosize=True,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        width=750,
        height=500,
    )

    return distPlot




def paretoChart(perioddata, typeofChart):
    keyHolder = findvalueHolder(perioddata)[0]
    valueHolder = findvalueHolder(perioddata)[1]
    countHolder = findvalueHolder(perioddata)[2]
    
    if typeofChart == 'time':
        timeTable = pd.DataFrame({'Sum': valueHolder})
        timeTable.index = keyHolder
        title ='Total Downtime for each Module'
    elif typeofChart == 'counts':
        timeTable = pd.DataFrame({'Sum': countHolder})
        timeTable.index = keyHolder
        title ='Alarm Frequency for each Module'
    
    #sort DataFrame by count descending
    timeTable = timeTable.sort_values(by='Sum', ascending=False)

    #add column to display cumulative percentage and use it to calculate difference
    timeTable['Cumulative Percentage'] = timeTable['Sum'].cumsum()/timeTable['Sum'].sum()*100
    
    timeTable['Percentage caused by each module'] = timeTable['Cumulative Percentage'].diff().round(2)
    
    timeTable['Percentage caused by each module'][0] = timeTable['Cumulative Percentage'][0].round(2)
    
    timeTable.drop("Cumulative Percentage", axis=1, inplace=True)
    
    paretoChart = go.Figure()

    paretoChart.add_trace(
        go.Bar(name= 'Total '+ typeofChart ,x= timeTable.index, y = timeTable["Sum"], marker_color = '#5DADEC', text = timeTable['Sum'], textfont=dict(color='#F5F5F5', size= 28)))

    paretoChart.add_trace(
            go.Scatter(
                name = "% caused by each module",
                # input all the names
                x = timeTable.index,
                # input the data for that column
                y = [1.0] * len(timeTable['Percentage caused by each module']),
                text = timeTable['Percentage caused by each module'],
                textposition="middle center", mode="text",
                marker=dict(color='#1B1811', size=4),textfont=dict(color='#C41E3A', size= 28)
            )
    )
    
    paretoChart.update_layout(
        title_text = title,
        width=600,
        height=400,
        margin=dict(l=0,r=0,b=0,t=25),
        paper_bgcolor="#FFFFFF",
    )
    
    paretoChart.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="left",
    x=0.6
    ))
    
    morethan50 = []
    morethan50 = timeTable.index.tolist()
    morethan50modules = morethan50[:3]
    morethan50perc = timeTable['Percentage caused by each module'][:3]
    
    output = ''
    for i in range(3):
        output += str(morethan50modules[i]) + ' - ' + str(morethan50perc[i]) + '%'
        output += '\n\n'
    
    return [paretoChart, output, timeTable]

def componentTable(perioddata, format, filter, prevdata):
    
    # filtering data and getting groupby component alarm
    filteredDf = perioddata.loc[perioddata['Module'] == filter]
    groupbydf = filteredDf.copy()
    groupbydf = groupbydf.groupby(['DT Reason Detail'])
    
    # GETTING prev data to do comparison
    prevfilteredDf = prevdata.loc[prevdata['Module'] == filter]
    groupbyprevdf = prevfilteredDf.copy()
    groupbyprevdf = groupbyprevdf.groupby(['DT Reason Detail'])
    
    componentDict = []

    for key, item in groupbydf:
        totaltime = sum(item['Duration(mins)'])
        componentDict.append([key, totaltime, len(item), (totaltime/ len(item))])
        
    prevcomponentDict = []

    for key, item in groupbyprevdf:
        totaltime = sum(item['Duration(mins)'])
        prevcomponentDict.append([key, (totaltime/ len(item))])
        
    for i in range(len(componentDict)):
        for subkey in prevcomponentDict:
            if subkey[0] == componentDict[i][0]:
                componentDict[i].append(subkey[1])

    componentDict = pd.DataFrame(columns = ['Component', 'Downtime', 'Count', 'Average Downtime per Alarm', 'Prev average'], data = componentDict)
    componentDict = componentDict.sort_values('Downtime', ascending=False)
    
    component_fig = go.Figure(data=[go.Table(
        columnwidth = [3,1,1,1,1],
        header=dict(values= ['Component', 'DT', 'Freq', 'Avg DT', 'Prev Mth Avg DT'],
                    fill_color= '#5DADEC',
                    align='left'),
        cells=dict(values=[componentDict.iloc[:, 0], componentDict.iloc[:, 1],
                           componentDict.iloc[:, 2], round(componentDict.iloc[:, 3], 1), round(componentDict.iloc[:, 4], 1)],
                fill_color='lavender',
                align='left'))
    ])
    
    component_fig.update_layout(
        autosize=True,
        height=265,
        width=470,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    component_fig.update_traces(cells_font=dict(size = 16))    

    return component_fig

def componenttimeChart(perioddata, dateDict, filter, format):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == filter]
    
    timeDict = {}
    
    timeDict.update({filter: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[row[5]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
    timeDist = go.Figure(layout = format)
    value = timeDict[filter]
    rollingavg = value.copy()
    rollingavg = pd.DataFrame([rollingavg]).T
    
    rollingavg[ '7day_rolling_avg' ] = rollingavg[0].rolling(7).mean()
            
    timeDist.add_trace(
            go.Scatter(
                # input all the names
                x = list(value.keys()),
                # input the data for that column
                y = list(value.values()),
                name = filter,
                visible = True,
                mode="lines+text",
                text = ["" if v == 0 else v for v in list(value.values())],
                textposition="top center",
            )
        )
    
    timeDist.add_trace(
        go.Scatter(
            name = '7-day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )          
    
    timeDist.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.4
                ))
        
    timeDist.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Total Downtime",
        width=750,
        height=500,
    
    )
    
    return timeDist


def componentcountChart(perioddata, dateDict, filter, format):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == filter]
    
    alarmDict = {}
    
    alarmDict.update({filter: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        alarmDict[row[5]][row[0].strftime("%#d/%#m/%Y")] += 1
    
    countDist = go.Figure(layout = format)
    value = alarmDict[filter]
    rollingavg = value.copy()
    rollingavg = pd.DataFrame([rollingavg]).T
    
    rollingavg[ '7day_rolling_avg' ] = rollingavg[0].rolling(7).mean()
    
    countDist.add_trace(
            go.Scatter(
                # input all the names
                x = list(value.keys()),
                # input the data for that column
                y = list(value.values()),
                name = filter,
                visible = True,
                mode="lines+text",
                text = ["" if v == 0 else v for v in list(value.values())],
                textposition="top center",
            )
        )
    
    countDist.add_trace(
        go.Scatter(
            name = '7-day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )          
    
    countDist.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.25
                ))
        
    countDist.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Alarm Frequency",
        width=750,
        height=500
    )
    
    return countDist

def moduledistributionChart(perioddata, dateDict, module):
    
    filteredDf = perioddata.loc[perioddata['Module'] == module]
    
    timeDict = {}
    
    timeDict.update({module: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[module][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
        
    colors = ['rgb(30, 50, 100)']
    
    distPlot = ff.create_distplot([[i for i in list(timeDict[module].values())]], [module], colors=colors,
                            show_curve=True)
    
    distPlot.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.25
                ))
  
    distPlot.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Probability",
        width=750,
        height=500,
    )

    return distPlot


def componentdistributionChart(perioddata, dateDict, component):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component]
    # st.write(filteredDf)
    
    timeDict = {}
    
    timeDict.update({component: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[component][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
    colors = ['rgb(30, 50, 100)']

    distPlot = ff.create_distplot([[i for i in list(timeDict[component].values())]], [component], colors=colors,
                            show_curve=True)
    
    distPlot.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.25
                ))
    
    distPlot.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Probability",
        width=750,
        height=500,
    )

    return distPlot



def automationAvail(perioddata, dateDict):
    
    timeDict = {}
    
    timeDict.update(dateDict.copy())
    
    for row in perioddata.values.tolist():
        timeDict[row[0].strftime("%#d/%#m/%Y")] += float(row[4])
        
    return timeDict

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
def print(hello):
    st.write(hello)
        
def hasitimproved(perioddata):
    # t-test 5%
    improvedoutput = ''
    noimprovedoutput = ''
    for module, daydata in perioddata.items():
        prevdata = list(daydata.values())[:-30]
        datatotest = list(daydata.values())[len(list(daydata.values())) - 30:]
        
        # Filtering out the outliers
        zscores = stats.zscore(prevdata)
        noofmorethan3 = len([i for i in zscores if i >= 3])
        minoutlier = 0
        if noofmorethan3 > 0:
            # getting all outliers
            timeagainstoutliers = [a*b for a,b in zip(prevdata,
                                              [1 if i>=3 else 0 for i in zscores])]
            minoutlier = statistics.mean([i for i in timeagainstoutliers if i >= 3])
        
        fulldataholder = [i for i in prevdata if minoutlier >= i >= 0]
        last30daysavg = round(statistics.mean(datatotest),5)
        
        t_value,p_value=stats.ttest_1samp(fulldataholder, last30daysavg)
        
        one_tailed_p_value=float("{:.6f}".format(p_value/2))
        
        # printer('p-value for one tailed test is %f'%one_tailed_p_value)
        
        # printer(module)
        # printer(t_value)
        # printer(statistics.mean(prevdata))
        # printer(last30daysavg)
        # printer(p_value)
        
        alpha = 0.05
        if one_tailed_p_value<=alpha:
            percentage = abs(round(((last30daysavg - statistics.mean(prevdata))/ statistics.mean(prevdata)) * 100, 1))
            if statistics.mean(prevdata) > last30daysavg:
                improvedoutput += str(module) + "'s DT ⤵️ "
                improvedoutput += str(percentage) + '%'
                improvedoutput += '\n\n'
                
            else:
                noimprovedoutput += str(module) + "'s DT ⤴️ " 
                noimprovedoutput += str(percentage) + '%'
                noimprovedoutput += '\n\n'
                
    return [improvedoutput, noimprovedoutput]
        
        
def counttimeDate(perioddata, component_selected):
    perioddata = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
    
    alongtime = perioddata.values.tolist()
    hour = {}
    allhours = []
    for i in range(0,24):
        # stores the total time and count for that hour
        hour[i] = hour.get(i, [0,0])
    
    for i in alongtime:
        specifichr = i[2].hour
        hour[specifichr][0] += i[4]
        hour[specifichr][1] += 1 
    
    totaltimelist = []
    countlist = []
    
    for key,value in hour.items():
        totaltimelist.append(value[0])
        countlist.append(value[1])

    counttimeDist = make_subplots(specs=[[{"secondary_y": True}]], vertical_spacing = 0.0)

    
    counttimeDist.add_trace(
        go.Scatter(
            x = list(hour.keys()),
            y = totaltimelist,
            visible = True,
            name = 'Downtime (Top Left) Label',
            mode="lines+text",
            text = totaltimelist,
            textposition="top left",
        ), secondary_y = False
    )
    
    counttimeDist.add_trace(
        go.Scatter(
            x = list(hour.keys()),
            y = countlist,
            visible = True,
            name = 'Frequency (Bottom Right Label)',
            mode="lines+text",
            text = countlist,
            textposition="bottom right",
        ), secondary_y = True
    )
    
    counttimeDist.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=0.6
    ))
    
   
    
    counttimeDist.update_layout(
        autosize=True,
        margin=dict(l=0,r=0,b=0,t=0),
        width=900,
        height=450,
        xaxis_title="Time of Day",
        yaxis_title="Count",
        xaxis=dict(rangeslider=dict(visible=True),
                             type="linear"))
    
    counttimeDist.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Black')
    counttimeDist.update_yaxes(showline=True, linewidth=2, linecolor='black')    
    return (counttimeDist)
    
    # st.plotly_chart(countDist)  
    # st.plotly_chart(timeDist)
    
    # alarmselected is the rootcause selected here
    # component_selected is the alarm selected
    
def componentmeantime(perioddata, dateDict, component_selected, alarmSelected):
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
    filteredDf = filteredDf.loc[perioddata['Diagonstics'] == alarmSelected]
    
    timeDict = {}
    alarmDict = {}
    noofdays = len(dateDict)
    
    timeDict.update(dateDict.copy())
    alarmDict.update(dateDict.copy())
    
    for row in filteredDf.values.tolist():
        timeDict[row[0].strftime("%#d/%#m/%Y")] += float(row[4])
        if alarmDict[row[0].strftime("%#d/%#m/%Y")] == 0:
            alarmDict[row[0].strftime("%#d/%#m/%Y")] += 1
    
    timeList = list(timeDict.values())
    mean = statistics.mean(timeList)
    above50timefreq = round(noofdays/ len([i for i in timeList if int(i) > mean]),1)
    below50timefreq = round(noofdays/ len([i for i in timeList if int(i) <= mean]),1)
    
    
    st.info(str(below50timefreq) + ' days ' + ' ⋞ DT 50%tile ' + ' (' + str(round(mean,2)) + ') ⋟ ' + str(above50timefreq) + ' days')
    
    
def alarmsTable(perioddata, format, component_selected, alarmSelected, prevdata):
    # filtering data and getting groupby component alarm
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
    groupbydf = filteredDf.copy()
    groupbydf = groupbydf.groupby(['Diagonstics'])
    
    # GETTING prev data to do comparison
    prevfilteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
    groupbyprevdf = prevfilteredDf.copy()
    groupbyprevdf = groupbyprevdf.groupby(['Diagonstics'])
    
    componentDict = []
    # st.write(componentDict)

    for key, item in groupbydf:
        totaltime = sum(item['Duration(mins)'])
        componentDict.append([key, totaltime, len(item), (totaltime/ len(item))])
       
    # st.write(componentDict) 
    prevcomponentDict = []

    for key, item in groupbyprevdf:
        totaltime = sum(item['Duration(mins)'])
        prevcomponentDict.append([key, (totaltime/ len(item))])
    # st.write(prevcomponentDict) 
        
    # putting in the prev avg data into the same list as the current avg data
    for i in range(len(componentDict)):
        for subkey in prevcomponentDict:
            if subkey[0] == componentDict[i][0]:
                componentDict[i].append(subkey[1])

    componentDict = pd.DataFrame(columns = ['Alarm', 'Downtime', 'Count', 'Average Downtime per Fault', 'Prev average'], data = componentDict)
    componentDict = componentDict.sort_values('Downtime', ascending=False)
    
    component_fig = go.Figure(data=[go.Table(
        columnwidth = [3,1,1,1,1],
        header=dict(values= ['Root Cause', 'DT', 'Freq', 'Avg DT', 'Prev Mth Avg DT'],
                    fill_color= '#5DADEC',
                    align='left'),
        cells=dict(values=[componentDict.iloc[:, 0], componentDict.iloc[:, 1],
                        componentDict.iloc[:, 2], round(componentDict.iloc[:, 3], 1), round(componentDict.iloc[:, 4], 1)],
                fill_color='lavender',
                align='left'))
    ])
    
    component_fig.update_layout(
        autosize=True,
        height=150,
        width=470,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    component_fig.update_traces(cells_font=dict(size = 14))    

    return component_fig


def faulttimeChart(perioddata, dateDict, DT, alarm, format):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == DT]
    filteredDf = filteredDf.loc[perioddata['Diagonstics'] == alarm]
    
    timeDict = {}
    
    timeDict.update({alarm: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[row[8]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
    timeDist = go.Figure(layout = format)
    value = timeDict[alarm]
    rollingavg = value.copy()
    rollingavg = pd.DataFrame([rollingavg]).T
    
    rollingavg[ '7day_rolling_avg' ] = rollingavg[0].rolling(7).mean()
            
    timeDist.add_trace(
            go.Scatter(
                # input all the names
                x = list(value.keys()),
                # input the data for that column
                y = list(value.values()),
                name = alarm,
                visible = True,
                mode="lines+text",
                text = ["" if v == 0 else v for v in list(value.values())],
                textposition="top center",
            )
        )
    
    timeDist.add_trace(
        go.Scatter(
            name = '7-day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )    
    
    timeDist.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.25
                ))      
        
    timeDist.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Total Downtime",
        width=750,
        height=500,
    )
    
    return timeDist


def faultcountChart(perioddata, dateDict, DT, alarm, format):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == DT]
    filteredDf = filteredDf.loc[perioddata['Diagonstics'] == alarm]
    
    alarmDict = {}
    
    alarmDict.update({alarm: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        alarmDict[row[8]][row[0].strftime("%#d/%#m/%Y")] += 1
    
    countDist = go.Figure(layout = format)
    value = alarmDict[alarm]
    rollingavg = value.copy()
    rollingavg = pd.DataFrame([rollingavg]).T
    
    rollingavg[ '7day_rolling_avg' ] = rollingavg[0].rolling(7).mean()
    
    countDist.add_trace(
            go.Scatter(
                # input all the names
                x = list(value.keys()),
                # input the data for that column
                y = list(value.values()),
                name = alarm,
                visible = True,
                mode="lines+text",
                text = ["" if v == 0 else v for v in list(value.values())],
                textposition="top center",
            )
        )
    
    countDist.add_trace(
        go.Scatter(
            name = '7-day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )          
    
    countDist.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.25
                ))
        
    countDist.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Alarm Frequency",
        width=750,
        height=500,
    )
    
    return countDist


def faultdistributionChart(perioddata, dateDict, DT , alarm):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == DT]
    filteredDf = filteredDf.loc[perioddata['Diagonstics'] == alarm]
    
    timeDict = {}
    
    timeDict.update({alarm: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[alarm][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
    colors = ['rgb(30, 50, 100)']

    distPlot = ff.create_distplot([[i for i in list(timeDict[alarm].values())]], [alarm], colors=colors,
                            show_curve=True)
    
    distPlot.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.25
                ))
    
    distPlot.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Probability",
        width=750,
        height=500,
        
    )

    return distPlot
