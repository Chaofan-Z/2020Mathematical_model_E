import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

path = "./"
amos1 = pd.read_csv(path+'amos20191216.csv')
amos1 = pd.read_csv(path+'amos20200313.csv')

amos1 = amos1.assign(year = lambda x:x['CREATEDATE'].map(lambda x:x[:4])). \
    assign(month = lambda x:x['CREATEDATE'].map(lambda x:x[5:7])). \
    assign(day = lambda x:x['CREATEDATE'].map(lambda x:x[8:10])). \
    assign(hour = lambda x:x['CREATEDATE'].map(lambda x:x.split(' ')[1].split(':')[0])). \
    assign(minute = lambda x:x['CREATEDATE'].map(lambda x:x.split(' ')[1].split(':')[1]))

# 按照小时对风速聚合
WS2A_mean = amos1.groupby(['hour'])['WS2A (MPS)'].mean().reset_index()
WS2A_mean.columns = ['hour','WS2A (MPS)_mean']
amos1 = pd.merge(amos1,WS2A_mean,'left','hour')

# 把数据规整到同一纬度
min_max_scaler = preprocessing.MinMaxScaler((0,100))
part_data = amos1[['QFE R06 (HPA)','QFE R24 (HPA)','QNH AERODROME (HPA)','RH (%)']]
amos1 = amos1.drop(['QFE R06 (HPA)','QFE R24 (HPA)','QNH AERODROME (HPA)','RH (%)'],1)
# part_data = part_data.T
part_data = min_max_scaler.fit_transform(part_data)
# part_data = part_data.T
part_data = pd.DataFrame(part_data,columns=['QFE R06 (HPA)','QFE R24 (HPA)','QNH AERODROME (HPA)','RH (%)'])
amos1 = pd.concat([amos1,part_data],1)

amos1.to_csv('./amos0313.csv',index=None)

for col in ['QFE R06 (HPA)','QFE R24 (HPA)','QNH AERODROME (HPA)','RH (%)','WS2A (MPS)_mean']:
    plt.figure(figsize=(5, 4), dpi=100)
    plt.plot(amos1[col])
    plt.title(col)
    plt.show()
