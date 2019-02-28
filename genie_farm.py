import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
df=pd.read_csv("tn.csv");
dist = input("Enter the District Id: ")
dist_id=int(dist)
crop = input("Enter the Crop Id: ")
crop_id=int(crop)
area = input("Enter the Land Area: ")
area=int(area)
inp=list()
out=list()
i=0
for i in range(0,13315):
    tt=int(df.iloc[i]['District_Id'])
    t=int(df.iloc[i]['Crop_Id'])
    if t == crop_id and tt == dist_id:
        inp.append([df.iloc[i]['Area']])
        out.append([df.iloc[i]['Production']])
#print(inp)
#print(out)
predictor = LinearRegression(n_jobs = -1)
predictor.fit(X=inp, y=out)
test = [[area]]
X=test
outcome = predictor.predict(X)

print('\n\nProduction : {}'.format(outcome))
x=inp
y=out
mean_x=np.mean(x)
mean_y=np.mean(y)
n=len(x)
numer=0
denom=0
for i in range(n):
    numer+=(x[i]-mean_x)*(y[i]-mean_y)
    denom+=(x[i]-mean_x)**2
b1=numer/denom
b0=mean_y -(b1*mean_x)    
ss_t=0
ss_r=0
for i in range(n):
    y_pred=b0+b1*x[i]
    ss_t+=(y[i]-mean_y)**2
    ss_r+=(y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print('\n\nRegression : {}'.format(r2))
max_x=np.max(x)+100
min_x=np.min(x)-100

x=np.linspace(min_x,max_x,1000)
y=b0+b1*x
plt.plot(x,y)
plt.scatter(inp, out,c='#ef5423',label='scatter plot')
plt.xlabel('Area')
plt.ylabel('production')
plt.legend()
plt.show()
