#!/usr/bin/env python
# coding: utf-8

# ### 加载数据集

# In[1]:


from utils import load_corpus, stopwords

TRAIN_PATH = "./data/weibo/train.txt"
TEST_PATH = "./data/weibo/test.txt"


# In[2]:


# 分别加载训练集和测试集
train_data = load_corpus(TRAIN_PATH)
test_data = load_corpus(TEST_PATH)


# In[3]:


import pandas as pd

df_train = pd.DataFrame(train_data, columns=["words", "label"])
df_test = pd.DataFrame(test_data, columns=["words", "label"])
df_train.head()


# ### 特征编码（词袋模型）

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern='\[?\w+\]?', 
                             stop_words=stopwords)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]


# In[5]:


X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]


# ### 训练模型&测试

# In[6]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)


# In[7]:


# 在测试集上用模型预测结果
y_pred = clf.predict(X_test)


# In[8]:


# 测试集效果检验
from sklearn import metrics

print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))


# ### 手动输入句子，判断情感倾向

# In[9]:


from utils import processing

strs = ["终于收获一个最好消息", "哭了, 今天怎么这么倒霉"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)


# In[10]:


output = clf.predict(vec)
output


# In[ ]:




