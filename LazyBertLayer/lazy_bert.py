import torch
from torch import nn
from transformers import BertLayer, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertPooler
from lazy_module_list import LazyModuleList


class LazyBert(BertModel):
    def __init__(self, config, add_pooling_layer=True, max_loaded_layers=1, lazy_schedule="oldest"):
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = LazyBertEncoder(config, max_loaded_layers=max_loaded_layers, lazy_schedule=lazy_schedule)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

class LazyBertEncoder(BertEncoder):
    def __init__(self, config, max_loaded_layers=1, lazy_schedule="oldest"):
        nn.Module.__init__(self)
        self.config = config
        self.layer = LazyModuleList(
            modules_defs=[
                (BertLayer, (config,), {}) for i in range(config.num_hidden_layers)
            ],
            modules_checkpoints=self.config.filenames,
            max_instantied=max_loaded_layers,
            delete_schedule=lazy_schedule,
        )
        self.gradient_checkpointing = False

"""
# Old version of LazyBert

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
"""
