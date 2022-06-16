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
