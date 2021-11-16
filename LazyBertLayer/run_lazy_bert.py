from lazy_bert_layer import LazyBert
from transformers import BertTokenizer
import torch

configuration = torch.load('bert_config.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# for i in range(configuration.num_hidden_layers):
#     torch.save(model.encoder.layer[i].state_dict(), 'bert_layer_' + str(i) + '.pt')

# model_state_dict = model.state_dict()
# keys_to_delete = []
# for i in model_state_dict:
#     if "layer" in i:
#         keys_to_delete.append(i)
# for i in keys_to_delete:
#     del model_state_dict[i]

# torch.save(model_state_dict, 'bert_model.pt')

for i in range(configuration.num_hidden_layers):
    if not hasattr(configuration, 'filenames'):
        setattr(configuration, 'filenames', [])
    configuration.filenames.append('bert_layer_' + str(i) + '.pt')

model = LazyBert(configuration).cuda()
model.load_state_dict(torch.load('bert_model.pt'), strict=False) # Hacky: strict = False so that dummy parameters are ignored

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to("cuda")
import time
start = time.time()
output = model(**encoded_input)
output = model(**encoded_input)
output = model(**encoded_input)
output = model(**encoded_input)
end = time.time()
print((end - start)/4)