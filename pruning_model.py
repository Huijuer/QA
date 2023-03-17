import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import *

arr = []

# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Encoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size)
#
#     def forward(self, inputs, hidden):
#         # 将输入序列编码成状态向量
#         outputs, hidden = self.lstm(inputs, hidden)
#         return outputs, hidden
#
#
# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(hidden_size * 2, hidden_size)
#         self.v = nn.Parameter(torch.rand(hidden_size))
#
#     def forward(self, hidden, encoder_outputs):
#         # 计算注意力权重
#         max_len = encoder_outputs.size(0)
#         h = hidden.repeat(max_len, 1, 1).transpose(0, 1)
#         attn_energies = self.score(h, encoder_outputs)
#         return F.softmax(attn_energies, dim=1).unsqueeze(1)
#
#     def score(self, hidden, encoder_outputs):
#         # 计算注意力能量
#         energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
#         energy = energy.transpose(1, 2)
#         v = self.v.repeat(energy.size(0), 1).unsqueeze(1)
#         energy = torch.bmm(v, energy).squeeze(1)
#         return energy
#
#
# class Decoder(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(Decoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attention = Attention(self.hidden_size)
#         self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size * 2, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         # 解码生成输出序列
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = self.attention(hidden, encoder_outputs)
#         context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
#         context = context.transpose(0, 1)
#
#         output, hidden = self.gru(torch.cat([embedded, context], 2), hidden)
#         output = F.log_softmax(self.out(torch.cat([output, context], 2)), dim=1)
#         return output, hidden, attn_weights
#

class PruningModel(nn.Module):

    def __init__(self, rel2idx, idx2rel, ls):
        super(PruningModel, self).__init__()
        self.label_smoothing = ls
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel

        # self.roberta_pretrained_weights = 'hfl/chinese-roberta-wwm-ext'
        # self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights,output_hidden_states=True,
        #                              use_cache=False)
        self.roberta_pretrained_weights = 'hfl/chinese-macbert-base'
        self.roberta_model = BertModel.from_pretrained(self.roberta_pretrained_weights, output_hidden_states=True,
                                                       use_cache=False)

        self.lstm = nn.LSTM(input_size=768, hidden_size=256//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.roberta_dim = 768
        # self.mid1 = 512
        # self.mid2 = 512
        # self.mid3 = 256
        # self.mid4 = 256
        self.fcnn_dropout = torch.nn.Dropout(0.1)
        # self.lin1 = nn.Linear(self.roberta_dim, self.mid1)
        # self.lin2 = nn.Linear(self.mid1, self.mid2)
        # self.lin3 = nn.Linear(self.mid2, self.mid3)
        self.lin4 = nn.Linear(256, 128)
        self.hidden2rel = nn.Linear(128, len(self.rel2idx))
        print(len(self.rel2idx))

        self.loss = torch.nn.BCELoss(reduction='sum')

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def applyNonLinear(self, outputs):
        # outputs = self.fcnn_dropout(self.lin1(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin2(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin3(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin4(outputs))
        # outputs = F.relu(outputs)
        # print(outputs.shape)
        out = self.lin4(outputs)
        out = F.relu(out)
        outputs = self.hidden2rel(out)
        # outputs = self.hidden2rel_base(outputs)
        return outputs

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1, 0)
        cls_embedding = states[0]
        # print(cls_embedding)
        arr.append(cls_embedding)
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        # print(cls_embedding)
        print('--------')
        if arr.__len__()>4:
            print(arr)
        return question_embedding

    def forward(self, question_tokenized, attention_mask, rel_one_hot):
        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)##torch.Size([8, 768])
        lstm_out, _ = self.lstm(question_embedding)
        # print(lstm_out.shape)
        prediction = self.applyNonLinear(lstm_out)
        prediction = torch.sigmoid(prediction)
        actual = rel_one_hot
        # if self.label_smoothing:
        #     actual = ((1.0 - self.label_smoothing) * actual) + (1.0 / actual.size(1))
        # print(actual)
        loss = self.loss(prediction, actual)
        return loss

    def get_score_ranked(self, question_tokenized, attention_mask):
        question_embedding = self.getQuestionEmbedding(question_tokenized.unsqueeze(0), attention_mask.unsqueeze(0))
        lstm_out, _ = self.lstm(question_embedding)
        prediction = self.applyNonLinear(lstm_out)
        # print(question_embedding[0][0])
        # print(question_embedding[0][200])
        # print(question_embedding[0][400])
        # print(question_embedding[0][80])
        # print(question_embedding[0][145])
        # print("========================",question_embedding.size())
        # print(question_embedding,"question_embedding")
        prediction = torch.sigmoid(prediction).squeeze()
        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return prediction


