from flask import Flask, request,  render_template
import os

# bert_lstm
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from utils import load_corpus_bert
import pandas as pd
from torch.utils.data import Dataset, DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./model/chinese_wwm_pytorch"
batch_size = 100

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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x)
        lstm_out_cat = torch.cat([h_n[-2], h_n[-1]], 1)
        out = self.fc(lstm_out_cat)
        out = self.sigmoid(out)
        return out

from sklearn import metrics

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
bert = BertModel.from_pretrained(MODEL_PATH).to(device)
net1 = torch.load("./model/bert_lstm_dnn_7.model")

# bert
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out
net2 = torch.load("./model/bert_dnn_8.model")
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    

@app.route('/sentence_pred')
def sentence_predict():
    outputs= ''
    sentence = request.args.get('sentence')
    sentence = [sentence]
    if request.args.get('model') == 'BERT_LSTM':
        tokens = tokenizer(sentence, padding=True)
        input_ids = torch.tensor(tokens["input_ids"]).to(device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(device)

        last_hidden_states = bert(input_ids, attention_mask=attention_mask)
        bert_output = last_hidden_states[0]
        outputs = net1(bert_output)
        outputs = outputs.tolist()
    elif request.args.get('model') == 'BERT':
        tokens = tokenizer(sentence, padding=True)
        input_ids = torch.tensor(tokens["input_ids"]).to(device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(device)

        last_hidden_states = bert(input_ids, attention_mask=attention_mask)
        bert_output = last_hidden_states[0][:, 0]
        outputs = net2(bert_output)
        outputs = outputs.tolist()
        print(outputs)
    return outputs

@app.route('/file_pred', methods=['GET','POST'])
def file_predict():
    if request.method == 'POST':
        # 接收文件
        f = request.files['file']
        print(request.files)
        f.save('upload/' + f.filename)
        # BERT_LSTM处理
        TEST_PATH = "./upload/" + f.filename
        test_data = load_corpus_bert(TEST_PATH)
        df_test = pd.DataFrame(test_data, columns=["text", "label"])
        test_data = MyDataset(df_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        y_pred, y_true = [], []
        with torch.no_grad():
            for words, labels in test_loader:
                tokens = tokenizer(words, padding=True)
                input_ids = torch.tensor(tokens["input_ids"]).to(device)
                attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
                last_hidden_states = bert(input_ids, attention_mask=attention_mask)
                bert_output = last_hidden_states[0]
                outputs = net1(bert_output)          # 前向传播
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
        return metrics.classification_report(y_true, y_pred).split()
    else:
        return "POST is allowed!"

if __name__=='__main__':
    app.run(host='0.0.0.0', port=80)
