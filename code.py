import pandas as pd
import numpy as np
import category_encoders as ce
import os
import gc
from tqdm import *
# 核心模型使用第三方库
import lightgbm as lgb
# 交叉验证所使用的第三方库
from sklearn.model_selection import StratifiedKFold, KFold
# 评估指标所使用的的第三方库
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import copy
import datetime
from chinese_calendar import is_workday
from numpy import nan
# 忽略报警所使用的第三方库
import pandas.util.testing as tm
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_row', 1000)


train_data = pd.read_csv("../data/raw_data/train.csv")
test_data = pd.read_csv("../data/raw_data/evaluation_public.csv")

train_data['istest'] = 0
test_data['istest'] = 1

data = pd.concat([train_data, test_data]).reset_index(drop=True)

data['id_by_me'] = pd.Series(range(len(data)))

del data['ip_type']

# 特征工程

#风险系数高的情况

data['is_forei/unknown'] = data['op_city'].apply(lambda x: 1 if x in['国外','未知'] else 0)

data['is_fail_code']=data['http_status_code'].apply(lambda x:0 if x==200 else 1)

data['is_code_5']=data['http_status_code'].apply(lambda x:1 if x in [500,502] else 0)

data['is_login_url'] = data['url'].apply(lambda x:1 if x in ['xxx.com/getVerifyCode','xxx.com/getLoginType'] else 0)





data['op_datetime'] = pd.to_datetime(data['op_datetime'])
data['day']=data['op_datetime'].astype(str).apply(lambda x:str(x)[5:10])
data['hour'] = data['op_datetime'].dt.hour
data['weekday'] = data['op_datetime'].dt.weekday+1

data = data.sort_values(by=['user_name', 'op_datetime']).reset_index(drop=True)

data['hour_sin'] = np.sin(data['hour']/24*2*np.pi)
data['hour_cos'] = np.cos(data['hour']/24*2*np.pi)

data['op_day'] = data['op_datetime'].astype(str).apply(lambda x:str(x)[8:10])

data['min'] = data['op_datetime'].apply(lambda x: int(str(x)[-5:-3]))
data['min_sin'] = np.sin(data['min']/60*2*np.pi)
data['min_cos'] = np.cos(data['min']/60*2*np.pi)

#user上次点击时间差
data['diff_last_1'] = data.groupby('user_name')['op_datetime'].apply(lambda i:i.diff(1)).dt.total_seconds()/60
data['diff_last_2'] = data.groupby('user_name')['op_datetime'].apply(lambda i:i.diff(2)).dt.total_seconds()/60

train_data = data[data['istest']==0]
test = data[data['istest']==1]

#train：
train_data['diff_next'] = -(train_data.groupby('user_name')['op_datetime'].apply(lambda i:i.diff(-1))).dt.total_seconds()/60
data=pd.merge(data,train_data[['diff_next','id_by_me']],how='left', on='id_by_me')

fea = ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser',
          'os_type', 'os_version',  'op_city', 'log_system_transform', 'url']

for col in fea:
    data[col+'_diff1_mean'] = data.groupby(col)['diff_last_1'].transform('mean')
    data[col+'_diff1_std'] = data.groupby(col)['diff_last_1'].transform('std')
    data[col+'_diff1_max'] = data.groupby(col)['diff_last_1'].transform('max')
    data[col+'_diff1_min'] = data.groupby(col)['diff_last_1'].transform('min')

for col in fea:
    data[col+'_diff_next_mean'] = data.groupby(col)['diff_next'].transform('mean')
    data[col+'_diff_next_std'] = data.groupby(col)['diff_next'].transform('std')
    data[col+'_diff_next_max'] = data.groupby(col)['diff_next'].transform('max')
    data[col+'_diff_next_min'] = data.groupby(col)['diff_next'].transform('min')

#data.drop(['browser_version_diff1_min','browser_diff1_min','os_type_diff1_min','os_version_diff1_min'],axis=1,inplace=True)

data=data.fillna(-999)



def is_fail_code(x):
    if x==200:
        return 0
    else:
        return 1

#data['is_fail_code']=data['http_status_code'].apply(is_fail_code)

data['is_fail_usr']=data['user_name'].apply(lambda x:1 if x == "-999" else 0)#登陆失败

def isweekend(x):
    if(x<6):
        return 0
    else:
        return 1

data['isweekend']=data['weekday'].apply(isweekend)

def isnight(x):
    if (x>7)and(x<20):
        return 0
    else:
        return 1

data['isnight'] = data['hour'].apply(isnight)

#节假日放假
holiday = ['01-31','02-01', '02-02', '02-03', '02-04', '02-05', '02-06',
           '04-03','04-04', '04-05', '05-01', '05-02', '05-03', '05-04',
           '06-03', '06-04', '06-05']
def if_holiday(x):
    if x in holiday:
        return 1
    else:
        return 0
data['isholiday'] = data['op_datetime'].apply(lambda x:if_holiday(str(x)[5:10]))

#调休
adjust = ['01-29', '01-30','04-02', '04-24','05-07']
def if_adjust(x):
    if x in adjust:
        return 1
    else:
        return 0
data['is_adjust'] = data['day'].apply(if_adjust)

data['is_not_work'] = data['isweekend'].astype(bool)&(~data['is_adjust'])|(data['isholiday'].astype(bool))



time_fea=['hour','weekday','min','isnight','isholiday','is_not_work']

for col in time_fea:
    data[col+'_diff1_mean_u'] = data.groupby(['user_name',col])['diff_last_1'].transform('mean')
    data[col+'_diff1_std_u'] = data.groupby(['user_name',col])['diff_last_1'].transform('std')

for col in time_fea:
    data[col+'_diff1_next_mean_u'] = data.groupby(['user_name',col])['diff_next'].transform('mean')
    data[col+'_diff1_next_std_u'] = data.groupby(['user_name',col])['diff_next'].transform('std')

del data['diff_next']




cols = ['id_by_me','user_name','ip_transform', 'device_num_transform',
       'browser_version', 'browser', 'os_type', 'os_version','http_status_code','op_city',
        'log_system_transform','url','op_datetime']

tmp=data[cols]

tmp['op_day'] = tmp['op_datetime'].dt.date

tmp = tmp.groupby(['user_name','op_day'],as_index=False).agg({'id_by_me':list,'ip_transform':list, 'device_num_transform':list,
       'browser_version':list, 'browser':list, 'os_type':list, 'os_version':list,'http_status_code':list,'op_city':list,
        'log_system_transform':list,'url':list})

def get_which_time(col_unique,fea):
    fea_dict = dict.fromkeys(col_unique,0)
    count_list=[]
    for i in range(len(fea)):
        fea_dict[fea[i]] = fea_dict[fea[i]]+1
        count_list.append(fea_dict[fea[i]])
    return count_list

for col in tqdm(['ip_transform', 'device_num_transform',
       'browser_version', 'browser', 'os_type', 'os_version','http_status_code','op_city',
        'log_system_transform','url']):
    col_unique=data[col].unique()
    tmp[col+'_countls'] = tmp[col].apply(lambda x:get_which_time(col_unique,x))

tmp=tmp.explode(['id_by_me', 'ip_transform',
       'device_num_transform', 'browser_version', 'browser', 'os_type',
       'os_version', 'http_status_code', 'op_city', 'log_system_transform',
       'url', 'ip_transform_countls',
       'device_num_transform_countls', 'browser_version_countls',
       'browser_countls', 'os_type_countls', 'os_version_countls',
       'http_status_code_countls', 'op_city_countls',
       'log_system_transform_countls', 'url_countls'])

tmp = tmp.reset_index(drop=True)

cols=['id_by_me','ip_transform_countls', 'device_num_transform_countls',
       'browser_version_countls', 'browser_countls', 'os_type_countls',
       'os_version_countls', 'http_status_code_countls', 'op_city_countls',
       'log_system_transform_countls', 'url_countls']

data=pd.merge(data,tmp[cols],on='id_by_me',how='left')

for col in ['ip_transform_countls', 'device_num_transform_countls',
       'browser_version_countls', 'browser_countls', 'os_type_countls',
       'os_version_countls', 'http_status_code_countls', 'op_city_countls',
       'log_system_transform_countls', 'url_countls']:
    data[col] = data[col].astype(int)






cols = ['id_by_me','user_name','ip_transform', 'device_num_transform','browser_version', 'browser', 'os_type', 
                 'os_version','http_status_code','op_city','log_system_transform','url','op_datetime']
tmp=data[cols]

#账号最近几次登陆
for x in range(1,30):
    tmp['usr_diff_last_'+str(x)] = tmp.groupby(['user_name'])['op_datetime'].apply(lambda i:i.diff(x)).dt.total_seconds()/60
merge_cols = [col for col in tmp.columns if '_diff_last_' in col]
tmp['ip_diff_list_30']=tmp[merge_cols].values.tolist()
tmp.drop(merge_cols,axis=1,inplace=True)

#账号最近几次登陆对应ip
for x in range(1,30):
    tmp['usr_last_ip'+str(x)] = tmp.groupby(['user_name'])['ip_transform'].apply(lambda i:i.shift(x))
merge_cols = [col for col in tmp.columns if '_last_' in col]
tmp['usr_ip_list_30']=tmp[merge_cols].values.tolist()
tmp.drop(merge_cols,axis=1,inplace=True)

def get_nunique_minute(diff_list,uni_list,minute):
    ls=[]
    for i in range(len(diff_list)):
        if diff_list[i]<minute:
            ls.append(uni_list[i])
        else:
            break
    return pd.Series(ls).nunique()

tmp['ip_time_nui_6'] = tmp.apply(lambda row:get_nunique_minute(row['ip_diff_list_30'],row['usr_ip_list_30'],60*6),axis=1)

tmp['ip_time_nui_12'] = tmp.apply(lambda row:get_nunique_minute(row['ip_diff_list_30'],row['usr_ip_list_30'],60*12),axis=1)

tmp['ip_time_nui_24'] = tmp.apply(lambda row:get_nunique_minute(row['ip_diff_list_30'],row['usr_ip_list_30'],60*24),axis=1)

cols=[col for col in tmp.columns if 'ip_time_nui_'in col]

cols.append('id_by_me')

data=pd.merge(data,tmp[cols],on='id_by_me',how='left')




cross_cols=[]

#department_city
data['department_op_city'] = data['department'].astype(str)+data['op_city'].astype(str)
cross_cols.append('department_op_city')

#department_log_system_transform
data['department_log_system_transform'] = data['department'].astype(str)+data['log_system_transform'].astype(str)
#data['department_log_system_transform'] = label_encoder(data['department_log_system_transform'])
cross_cols.append('department_log_system_transform')

#browser_version_op_city
data['browser_version_op_city'] = data['browser_version'].astype(str)+data['op_city'].astype(str)
#data['browser_version_op_city'] = label_encoder(data['browser_version_op_city'])
cross_cols.append('browser_version_op_city')

#browser_op_city
data['browser_op_city'] = data['browser'].astype(str)+data['op_city'].astype(str)
#data['browser_op_city'] = label_encoder(data['browser_op_city'])
cross_cols.append('browser_op_city')

#browser_log_system_transform
data['browser_log_system_transform'] = data['browser'].astype(str)+data['log_system_transform'].astype(str)
#data['browser_log_system_transform'] = label_encoder(data['browser_log_system_transform'])
cross_cols.append('browser_log_system_transform')

#os_type_op_city
data['os_type_op_city'] = data['os_type'].astype(str)+data['op_city'].astype(str)
#data['os_type_op_city'] = label_encoder(data['os_type_op_city'])
cross_cols.append('os_type_op_city')

#os_type_log_system_transform
data['os_type_log_system_transform'] = data['os_type'].astype(str)+data['log_system_transform'].astype(str)
#data['os_type_log_system_transform'] = label_encoder(data['os_type_log_system_transform'])
cross_cols.append('os_type_log_system_transform')

#os_version_op_city
data['os_version_op_city'] = data['os_version'].astype(str)+data['op_city'].astype(str)
#data['os_version_op_city'] = label_encoder(data['os_version_op_city'])
cross_cols.append('os_version_op_city')

#os_type_log_system_transform
data['os_type_log_system_transform'] = data['os_type'].astype(str)+data['log_system_transform'].astype(str)
#data['os_type_log_system_transform'] = label_encoder(data['os_type_log_system_transform'])
cross_cols.append('os_type_log_system_transform')

#op_city_log_system_transform
data['op_city_log_system_transform'] = data['op_city'].astype(str)+data['log_system_transform'].astype(str)
#data['op_city_log_system_transform'] = label_encoder(data['op_city_log_system_transform'])
cross_cols.append('op_city_log_system_transform')


#departmen_url
data['op_city_log_system_transform'] = data['department'].astype(str)+data['log_system_transform'].astype(str)
#data['op_city_log_system_transform'] = label_encoder(data['op_city_log_system_transform'])
cross_cols.append('op_city_log_system_transform')




cols = ['ip_transform', 'device_num_transform',
       'browser_version', 'browser', 'os_type', 'os_version',
       'http_status_code', 'op_city', 'log_system_transform', 'url']

for col in cols:
    tmp = data[data['istest']==0].groupby(['user_name',col,'hour'])['is_risk'].count().reset_index()
    tmp.columns=['user_name',col,'hour',col+'_hour_count']
    data=pd.merge(data,tmp,on=['user_name',col,'hour'],how='left')


tmp = data[data['istest']==0].groupby(['user_name','is_not_work','hour'],as_index=False)['is_risk'].agg({'work_hour_count':'count'})
data = pd.merge(data,tmp,how='left',on=['user_name','is_not_work','hour'])

date_fea = ['weekday','isholiday']

for col in date_fea:
    tmp = data[data['istest']==0].groupby(['user_name',col,'hour'],as_index=False)['is_risk'].agg({col+'_count':'count'})
    data = pd.merge(data,tmp,how='left',on=['user_name',col,'hour'])





numeric_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(include=[np.object])


data[categorical_features.columns] = ce.OrdinalEncoder().fit_transform(data[categorical_features.columns])







train_data = data[data['istest']==0]
test = data[data['istest']==1]

test = test.sort_values('id').reset_index(drop=True)

train = train_data[train_data['op_datetime']<'2022-04-01'].reset_index(drop=True)
val = train_data[train_data['op_datetime']>='2022-04-01'].reset_index(drop=True)

fea = [col for col in data.columns if col not in['id','index','id_by_me','op_datetime', 'op_day','day', 'op_month','is_risk', 'ts', 'ts1', 'ts2', 'diff_next']]

x_train = train[fea]
y_train = train['is_risk']

x_val = val[fea]
y_val = val['is_risk']

x_test = test[fea]
y_test = test['is_risk']



importance = 0
pred_y = pd.DataFrame()
var_pre = pd.DataFrame()

lgb_score_list = []

seeds=2022

params_lgb  = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 64,
    'verbose': -1,
    'seed': 2022,
    'n_jobs': -1,

    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 4,
    # 'min_child_weight': 10,
    "min_data_in_leaf":20
}


#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
kf = KFold(n_splits=5, shuffle=True, random_state=2022)
for i, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
    print('************************************ {} {}************************************'.format(str(i+1), str(seeds)))
    trn_x, trn_y, val_x, val_y = x_train.iloc[train_idx],y_train.iloc[train_idx], x_train.iloc[val_idx], y_train.iloc[val_idx]
    train_data = lgb.Dataset(trn_x,
                        trn_y)
    val_data = lgb.Dataset(val_x,
                        val_y)
    model = lgb.train(params_lgb, train_data, valid_sets=[val_data], num_boost_round=20000,
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])

    pred_y['fold_%d_seed_%d' % (i, seeds)] = model.predict(x_test)
    var_pre['fold_%d_seed_%d' % (i, seeds)] = model.predict(x_val)
    
    importance += model.feature_importance(importance_type='gain') / 5
    lgb_score_list.append(auc(val_y, model.predict(val_x)))

test['is_risk'] = pred_y.mean(axis=1).values

df_test = pd.read_csv('../data/raw_data/evaluation_public.csv')
df_test = pd.merge(df_test,test[['id','is_risk']],how='left')
df_test['op_datetime'] = pd.to_datetime(df_test['op_datetime'])
df_test = df_test.sort_values('op_datetime').reset_index(drop=True)
df_test['hour'] = df_test['op_datetime'].dt.hour
df_test.loc[df_test['hour']<6,'is_risk'] = 1
df_test.loc[df_test['hour']>20,'is_risk'] = 1

df_test[['id','is_risk']].to_csv("../data/prediction_result/result.csv", index=False)





