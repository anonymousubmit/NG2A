import numpy as np

filename = 'results10exp.log'

f = open(filename, 'r')
lines = f.readlines()
f.close()

dataDict = dict()

for line in lines:
    elements = line.strip('\n\r').split(' ')
    dataset = elements[0]
    seed = elements[1]
    result = eval(elements[2])
    if dataset not in dataDict:
        dataDict[dataset] = [result]
    else:
        dataDict[dataset].append(result)
    
print('datasets mean std')

for k, v in dataDict.items():
    v = np.array(v)
    mean_v = np.mean(v)
    std_v = np.std(v)
    print(k, mean_v, std_v)
