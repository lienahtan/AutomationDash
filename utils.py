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
                name = module_selected,
                visible = True,
                mode="lines+text",
                text = ["" if v == 0 else v for v in list(value.values())],
                textposition="top center",
            )
        )
    
    timeDist.add_trace(
        go.Scatter(
            name = '7 -day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )          
        
    timeDist.update_layout(
        autosize=True,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        width=750,
        height=500
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
            name = '7 -day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    ) 
        
    timeDistCount.update_layout(
        autosize=True,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        width=750,
        height=500
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
        height=500
    )

    return distPlot




def paretoChart(perioddata, typeofChart):
    keyHolder = findvalueHolder(perioddata)[0]
    valueHolder = findvalueHolder(perioddata)[1]
    countHolder = findvalueHolder(perioddata)[2]
    
    if typeofChart == 'time':
        timeTable = pd.DataFrame({'count': valueHolder})
        timeTable.index = keyHolder
        title ='Total time for each module'
    elif typeofChart == 'counts':
        timeTable = pd.DataFrame({'count': countHolder})
        timeTable.index = keyHolder
        title ='Alarm count for each module'
    
    #sort DataFrame by count descending
    timeTable = timeTable.sort_values(by='count', ascending=False)

    #add column to display cumulative percentage
    timeTable['cumperc'] = timeTable['count'].cumsum()/timeTable['count'].sum()*100
    
    paretoChart = make_subplots(specs=[[{"secondary_y": True}]], horizontal_spacing = 0.0)

    paretoChart.add_trace(
        go.Bar(name = "Modules", x= timeTable.index, y = timeTable["count"], marker_color = '#5DADEC'))
    
    paretoChart.add_trace(
            go.Scatter(
                name = "%",
                # input all the names
                x = timeTable.index,
                # input the data for that column
                y = (timeTable["cumperc"]/100) * 100,
                visible = True,
            ), secondary_y= True
    )
    
    paretoChart.update_layout(
        title_text = title,
        width=600,
        height=400,
        margin=dict(l=0,r=0,b=0,t=25),
        paper_bgcolor="#FFFFFF",
    )
    morethan50 = []
    morethan50 = timeTable.index[timeTable['cumperc']<50].tolist()
    
    return [paretoChart, morethan50, timeTable]

def componentTable(perioddata, format, filter):
    
    filteredDf = perioddata.loc[perioddata['Module'] == filter]
    groupbydf = filteredDf.copy()
    groupbydf = groupbydf.groupby(['DT Reason Detail'])
    
    componentDict = []

    for key, item in groupbydf:
        totaltime = sum(item['Duration(mins)'])
        componentDict.append([key, totaltime, len(item), (totaltime/ len(item))])
        
    componentDict = pd.DataFrame(columns = ['Component', 'Downtime', 'Count', 'Average Downtime per Alarm'], data = componentDict)
    componentDict = componentDict.sort_values('Downtime', ascending=False)
    
    component_fig = go.Figure(data=[go.Table(
        columnwidth = [1.5,1,1,1],
        header=dict(values= ['Component', 'Downtime', 'Count', 'Downtime/ Alarm'],
                    fill_color= '#5DADEC',
                    align='left'),
        cells=dict(values=[componentDict.iloc[:, 0], componentDict.iloc[:, 1],
                           componentDict.iloc[:, 2], round(componentDict.iloc[:, 3], 1)],
                fill_color='lavender',
                align='left'))
    ])
    
    component_fig.update_layout(
        autosize=True,
        height=200,
        width=470,
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return component_fig

def componenttimeChart(perioddata, dateDict, filter, format):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == filter]
    
    timeDict = {}
    alarmDict = {}
    
    timeDict.update({filter: dateDict.copy()})
    alarmDict.update({filter: dateDict.copy()})
    
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
            name = '7 -day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )          
        
    timeDist.update_layout(
        autosize=True,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        width=750,
        height=500
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
            name = '7 -day moving average',
            x = list(value.keys()),
            y = rollingavg['7day_rolling_avg' ],
            visible= True,
            mode = "lines"
        )
    )          
        
    countDist.update_layout(
        autosize=True,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        width=750,
        height=500
    )
    
    return countDist


def componentdistributionChart(perioddata, dateDict, component):
    
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component]
    
    timeDict = {}
    
    timeDict.update({component: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[row[5]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
    
    # distPlot = go.Figure(layout= format)
    value = timeDict[component]
    colors = ['rgb(0, 10, 100)']
    
    distPlot = ff.create_distplot([[i for i in list(value.values()) if i != 0]], [component], colors=colors,
                            show_curve=True)
    
  
    distPlot.update_layout(
        autosize=True,
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title",
        width=750,
        height=500
    )

    return distPlot

