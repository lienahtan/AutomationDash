import pandas as pd

# creating range of all dates from data
def get_dateDict(startDate, noofdays):
    dateDict = {}
    datelist = pd.date_range(startDate, periods=noofdays + 1)
    indexDate = []
    
    for date in datelist:
        indexDate.append(date.strftime("%#d/%#m/%Y"))
    
    for date in indexDate:
        dateDict.update({date: 0})
        
    return dateDict


