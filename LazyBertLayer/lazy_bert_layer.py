import torch
from torch import nn
from transformers import BertLayer, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertPooler


class LazyBert(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = LazyBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

class LazyBertEncoder(BertEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList([LazyBertLayer(config, i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

class LazyBertLayer(nn.Module):
    def __init__(self, config, layer_number: int):
        super(LazyBertLayer, self).__init__()
        self.dummy_parameter = nn.Parameter(torch.Tensor([1]))
        self.config = config
        self.layer_number = layer_number

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        layer = BertLayer(self.config).to(self.dummy_parameter.device)
        layer.load_state_dict(torch.load(self.config.filenames[self.layer_number]))
        ret = layer(*args, **kwargs)
        del layer
        return ret

