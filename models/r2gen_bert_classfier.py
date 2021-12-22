import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertClassfier(nn.Module):
    def __init__(self, bert_base_model, out_dim, freeze_layers):
        super(BertClassfier, self).__init__()
        # init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model, freeze_layers)
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(768, 768)  # 768 is the size of the BERT embbedings
        self.bert_l2 = nn.Linear(768, out_dim)  # 768 is the size of the BERT embbedings


    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = BertModel.from_pretrained(bert_model_name)  # , return_dict=True)
            # print("Image feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, encoded_inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        # print("encoded_inputs", encoded_inputs)
        outputs = self.bert_model(**encoded_inputs)
        # print("text_encoder outputs")
        # print(outputs)

        with torch.no_grad():
            v = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
            sentence_embeddings = v.to(dtype=torch.float16)
            # print(v.dtype)
            # print(sentence_embeddings.dtype)
            # sentence_embeddings = v.half()
            x = self.bert_l1(v)
            x = F.relu(x)
            out_emb = self.bert_l2(x)

        return out_emb

