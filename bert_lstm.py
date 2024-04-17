#!/usr/bin/env python
# coding: utf-8

# ### 加载数据集

# In[1]:


from utils import load_corpus_bert

TRAIN_PATH = "./data/weibo/train.txt"
TEST_PATH = "./data/weibo/test.txt"


# In[2]:


# 分别加载训练集和测试集
train_data = load_corpus_bert(TRAIN_PATH)
test_data = load_corpus_bert(TEST_PATH)


# In[3]:


import pandas as pd

df_train = pd.DataFrame(train_data, columns=["text", "label"])
df_test = pd.DataFrame(test_data, columns=["text", "label"])
df_train.head()


# ### 加载bert

# In[4]:


import os
from transformers import BertTokenizer, BertModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    # 在我的电脑上不加这一句, bert模型会报错
MODEL_PATH = "./model/chinese_wwm_pytorch"     # 下载地址见 https://github.com/ymcui/Chinese-BERT-wwm


# In[5]:


# 加载
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)   # 分词器
bert = BertModel.from_pretrained(MODEL_PATH)            # 模型


# ### 神经网络

# In[6]:


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"

bert = bert.to(device)


# In[7]:


# 超参数
learning_rate = 1e-3
input_size = 768
num_epoches = 10
batch_size = 100
decay_rate = 0.9
hidden_size = 64
num_layers = 2


# In[8]:


# 数据集
class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df["text"].tolist()
        self.label = df["label"].tolist()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)

# 训练集
train_data = MyDataset(df_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 测试集
test_data = MyDataset(df_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[9]:


# 网络结构
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向, 输出维度要*2
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x)
        # print(len(lstm_out))
        # print(h_n[-1].size())
        # print(h_n[-2].size())
        lstm_out_cat = torch.cat([h_n[-2], h_n[-1]], 1)  # 双向, 所以要将最后两维拼接, 得到的就是最后一个time step的输出
        out = self.fc(lstm_out_cat)
        out = self.sigmoid(out)
        return out

lstm = LSTM(input_size, hidden_size, num_layers).to(device)


# In[10]:


from sklearn import metrics

# 测试集效果检验
def test():
    y_pred, y_true = [], []

    with torch.no_grad():
        for words, labels in test_loader:
            tokens = tokenizer(words, padding=True)
            input_ids = torch.tensor(tokens["input_ids"]).to(device)
            attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
            last_hidden_states = bert(input_ids, attention_mask=attention_mask)
            bert_output = last_hidden_states[0]
            outputs = lstm(bert_output)          # 前向传播
            outputs = outputs.view(-1)          # 将输出展平
            y_pred.append(outputs)
            y_true.append(labels)

    y_prob = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = y_prob.clone()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    y_prob = y_prob.cpu()
    print(metrics.classification_report(y_true, y_pred))
    print("准确率:", metrics.accuracy_score(y_true, y_pred))
    print("AUC:", metrics.roc_auc_score(y_true, y_prob) )


# In[11]:


# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)


# In[ ]:


# 迭代训练
for epoch in range(num_epoches):
    total_loss = 0
    for i, (words, labels) in enumerate(train_loader):
        tokens = tokenizer(words, padding=True)
        input_ids = torch.tensor(tokens["input_ids"]).to(device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            last_hidden_states = bert(input_ids, attention_mask=attention_mask)
            bert_output = last_hidden_states[0]
        optimizer.zero_grad()               # 梯度清零
        # print(bert_output.size()) 100*786
        outputs = lstm(bert_output)          # 前向传播
        logits = outputs.view(-1)           # 将输出展平
        loss = criterion(logits, labels)    # loss计算
        total_loss += loss
        loss.backward()                     # 反向传播，计算梯度
        optimizer.step()                    # 梯度更新
        if (i+1) % 10 == 0:
            print("epoch:{}, step:{}, loss:{}".format(epoch+1, i+1, total_loss/10))
            total_loss = 0
    
    # learning_rate decay
    scheduler.step()
    
    # test
    test()
    
    # save model
    model_path = "./model/bert_lstm_dnn_{}.model".format(epoch+1)
    torch.save(lstm, model_path)
    print("saved model: ", model_path)


# # 手动输入句子，判断情感倾向（1正/0负）

# In[12]:


net = torch.load("./model/bert_lstm_dnn_4.model")    # 训练过程中的巅峰时刻


# In[13]:


s = ["华丽繁荣的城市、充满回忆的小镇、郁郁葱葱的山谷...", "突然就觉得人间不值得"]
tokens = tokenizer(s, padding=True)
input_ids = torch.tensor(tokens["input_ids"]).to(device)
attention_mask = torch.tensor(tokens["attention_mask"]).to(device)

last_hidden_states = bert(input_ids, attention_mask=attention_mask)
bert_output = last_hidden_states[0]
outputs = net(bert_output)
outputs


# In[15]:


s = ["今天天气真好", "今天天气特别特别棒"]
tokens = tokenizer(s, padding=True)
input_ids = torch.tensor(tokens["input_ids"]).to(device)
attention_mask = torch.tensor(tokens["attention_mask"]).to(device)

last_hidden_states = bert(input_ids, attention_mask=attention_mask)
bert_output = last_hidden_states[0]
outputs = net(bert_output)
outputs


# In[ ]:




