import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF

from pytorch_transformers import BertPreTrainedModel, BertModel

class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config):
        super(BERT_BiLSTM_CRF, self).__init__()
        
        self.num_tags = config.num_tags
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size

        # 如果为False，则不要Bilstm层
        if config.need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, config.rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = config.rnn_dim*2
        
        self.hidden2tag = nn.Linear(out_dim, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)
    

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1*self.crf(emissions, tags, mask=input_mask.byte())

        return loss

    
    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions
    
    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())



