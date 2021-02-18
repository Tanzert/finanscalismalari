#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data = pd.read_excel(r'C:\Users\2019\Desktop\İŞ\python ile portföy optimizasyonu\portfoyopt.xlsx')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


symbols = ['onsalt', 'onsgms', 'usdtry']
df = data.set_index('date')


# In[14]:


onsaltin = df['onsalt'].pct_change().apply(lambda x: np.log(1+x))
onsaltin.head()


# In[15]:


var_onsalt = onsaltin.var()
var_onsalt


# In[16]:


onsgumus = df['onsgms'].pct_change().apply(lambda x: np.log(1+x))
onsgumus.head()


# In[17]:


var_onsgms = onsgumus.var()
var_onsgms


# In[18]:


usdtotry = df['usdtry'].pct_change().apply(lambda x: np.log(1+x))
usdtotry.head()


# In[19]:


var_usdtry = usdtotry.var()
var_usdtry


# In[20]:


onsalt_vol = np.sqrt(var_onsalt)
onsgms_vol = np.sqrt(var_onsgms)
usdtry_vol = np.sqrt(var_usdtry)
onsalt_vol, onsgms_vol, usdtry_vol


# In[22]:


test = df.pct_change().apply(lambda x: np.log(1+x))
test.head()


# In[25]:


test['onsalt'].cov(test['onsgms'])


# In[26]:


test['onsalt'].corr(test['onsgms'])


# In[27]:


test1 = df.pct_change().apply(lambda x: np.log(1+x))
test1.head()


# In[28]:


w = [0.3, 0.3, 0.4]
e_r_ind = test1.mean()
e_r_ind


# In[29]:


#beklenen toplam getiri
e_r = (e_r_ind*w).sum()
e_r


# In[30]:


#kovaryans matrisi
cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix


# In[31]:


#korelasyon matrisi
corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
corr_matrix


# In[32]:


w = {'onsalt': 0.3, 'onsgms': 0.3, 'usdtry': 0.4}
port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()
port_var


# In[33]:


ind_er = df.resample('Y').last().pct_change().mean()
ind_er


# In[34]:


#portföy getirileri
w = [0.3, 0.3, 0.4]
port_er = (w*ind_er).sum()
port_er


# In[35]:


ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(12))
ann_sd


# In[36]:


assets = pd.concat([ind_er, ann_sd], axis=1)
assets.columns = ['Getiriler', 'Oynaklık']
assets


# In[37]:


p_ret = []
p_vol = [] 
p_weights = [] 

num_assets = len(df.columns)
num_portfolios = 10000


# In[38]:


for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    sd = np.sqrt(var) 
    ann_sd = sd*np.sqrt(12) 
    p_vol.append(ann_sd)


# In[45]:


data = {'Getiriler':p_ret, 'Oynaklık':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):

    data[symbol+' weight'] = [w[counter] for w in p_weights]


# In[46]:


portfolios  = pd.DataFrame(data)
portfolios.head()


# In[49]:


portfolios.plot.scatter(x='Oynaklık', y='Getiriler', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])


# In[50]:


min_vol_port = portfolios.iloc[portfolios['Oynaklık'].idxmin()]                              
min_vol_port


# In[51]:


plt.subplots(figsize=[10,10])
plt.scatter(portfolios['Oynaklık'], portfolios['Getiriler'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)


# In[52]:


rf = 0.01 # risk faktörü
optimal_risky_port = portfolios.iloc[((portfolios['Getiriler']-rf)/portfolios['Oynaklık']).idxmax()]
optimal_risky_port


# In[53]:


plt.subplots(figsize=(10, 10)) #yeşil yıldız optimal riskli portföyü temsil ediyor
#kırmızı yıldız minimum oynaklıktaki en etkili portföyü temsil ediyor
plt.scatter(portfolios['Oynaklık'], portfolios['Getiriler'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)


# In[ ]:




