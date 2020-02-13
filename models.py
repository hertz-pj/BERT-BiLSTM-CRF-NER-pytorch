import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF

from pytorch_transformers import BertPreTrainedModel, BertModel

class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        
        self.num_tags = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim*2
        
        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
    

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1*self.crf(emissions, tags, mask=input_mask.byte())

        return loss

    
    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]
        
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions
    
    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())

