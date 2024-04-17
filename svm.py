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


# ### 特征编码（Tf-Idf模型）

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(token_pattern='\[?\w+\]?', 
                             stop_words=stopwords)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]


# In[5]:


X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]


# ### 训练模型&测试

# In[ ]:


from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)


# In[ ]:


# 在测试集上用模型预测结果
y_pred = clf.predict(X_test)


# In[ ]:


# 测试集效果检验
from sklearn import metrics

print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))


# ### 手动输入句子，判断情感倾向

# In[ ]:


from utils import processing

strs = ["只要流过的汗与泪都能化作往后的明亮，就值得你为自己喝彩", "烦死了！为什么周末还要加班[愤怒]"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)


# In[ ]:


output = clf.predict(vec)
output


# In[ ]:




