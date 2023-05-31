#!/usr/bin/env python
# coding: utf-8

# ### RFM-анализ

# In[20]:


import pandas as pd
import numpy as np
import datetime 

# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt

# We will use the Seaborn library
import seaborn as sns
sns.set()

# Graphics in SVG format are more sharp and legible
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['image.cmap'] = 'viridis'


# In[21]:


# Считываем данные
orders = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-a-/Segment_less4/RFM_ht_data.csv',sep=',')
orders


# In[22]:


# Преобразуем колонки, приведем к одному типу
orders['InvoiceDate'] = pd.to_datetime(orders['InvoiceDate'])
orders['InvoiceNo']=orders['InvoiceNo'].astype('str')
orders['CustomerCode']=orders['CustomerCode'].astype('str')


# In[23]:


orders.dtypes


# In[24]:


orders.CustomerCode.nunique()


# In[25]:


# last_date = datetime.date(2020,9, 30)
last_date =orders.InvoiceDate[332729]
last_date


# In[26]:


rfmTable = orders.groupby('CustomerCode').agg({'InvoiceDate': lambda x: (last_date - x.max()).days, # Recency #Количество дней с последнего заказа
                                        'InvoiceNo': lambda x: len(x),      # Frequency #Количество заказов
                                        'Amount': lambda x: x.sum()}) # Monetary Value #Общая сумма по всем заказам

rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'Amount': 'monetary_value'}, inplace=True)


# In[27]:


rfmTable


# In[9]:


rfmSegmentation = rfmTable


# In[10]:


def RClass(value,parameter_name,quantiles_table):
    if value <= quantiles_table[parameter_name][0.25]:
        return 1
    elif value <= quantiles_table[parameter_name][0.50]:
        return 2
    elif value <= quantiles_table[parameter_name][0.75]: 
        return 3
    else:
        return 4

def FMClass(value, parameter_name,quantiles_table):
    if value <= quantiles_table[parameter_name][0.25]:
        return 4
    elif value <= quantiles_table[parameter_name][0.50]:
        return 3
    elif value <= quantiles_table[parameter_name][0.75]: 
        return 2
    else:
        return 1


# In[11]:


quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles


# In[12]:



rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency', quantiles))

rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency', quantiles))

rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass, args=('monetary_value', quantiles))

rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str)                             + rfmSegmentation.F_Quartile.map(str)                             + rfmSegmentation.M_Quartile.map(str)


# In[13]:


rfmSegmentation


# In[14]:


rfmSegmentation.groupby('RFMClass').agg({'monetary_value':'count'}).sort_values('monetary_value')


# In[15]:


pd.crosstab(index = rfmSegmentation.R_Quartile, columns = rfmSegmentation.F_Quartile)


# In[28]:



rfm_table = rfmSegmentation.pivot_table(
                        index='R_Quartile', 
                        columns='F_Quartile', 
                        values='monetary_value', 
                        aggfunc=np.median).applymap(int)
sns.heatmap(rfm_table, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=4.15, annot_kws={"size": 10},yticklabels=4);


# In[48]:


rfmSegmentation=rfmSegmentation.reset_index()


# In[49]:


# Какая верхняя граница у суммы покупок у пользователей с классом 4 в подсегменте М? 
rfmSegmentation.query('M_Quartile ==  "4"').sort_values("monetary_value")


# In[50]:


# Какая нижняя граница у количества покупок у пользователей с классом 1 в подсегменте F?
rfmSegmentation.query('F_Quartile ==  "1"').sort_values("frequency").sort_values('frequency')


# In[51]:


# Какое максимальное количество дней может пройти с момента последней покупки для того, чтобы пользователь попал в класс 2 в подсегменте R?
rfmSegmentation.query('R_Quartile ==  "2"').sort_values("recency").sort_values('recency')


# In[52]:


# Сколько пользователей попало в сегмент 111?
rfmSegmentation.query('RFMClass ==  "111"').CustomerCode.nunique()


# In[42]:


rfmSegmentation.query('RFMClass ==  "444"').sort_values("monetary_value")


# In[31]:


rfm_table


# In[32]:


rfmTable


# In[33]:


rfmSegmentation.sort_values("frequency")


# In[53]:


# Сколько пользователей попало в сегмент 311?
rfmSegmentation[rfmSegmentation['RFMClass']=="311"].sort_values('recency', ascending=False).CustomerCode.nunique()


# In[35]:


rfmSegmentation1=rfmSegmentation.reset_index()


# In[39]:


# В каком RFM-сегменте самое большое кол-во пользователей
# В каком RFM-сегменте самое маленькое кол-во пользователей?
# Какое количество пользователей попало в самый малочисленный сегмент?
rfmSegmentation1.groupby('RFMClass').agg({'CustomerCode':'count'}).sort_values('CustomerCode', ascending=False)

