from transformers import BertModel, BertConfig
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer


def split_checkpoint(model, configuration):
    for i in range(configuration.num_hidden_layers):
        torch.save(model.encoder.layer[i].state_dict(), 'bert_layer_' + str(i) + '.pt')

    model_state_dict = model.state_dict()
    keys_to_delete = []
    for i in model_state_dict:
        if "layer" in i:
            keys_to_delete.append(i)
    for i in keys_to_delete:
        del model_state_dict[i]

    torch.save(model_state_dict, 'bert_model.pt')
    torch.save(configuration, 'bert_config.pt')

def benchmark(model):
    model.to("cuda")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt').to("cuda")
    import time
    start = time.time()
    with torch.no_grad():
        output = model(**encoded_input)
        output = model(**encoded_input)
        output = model(**encoded_input)
        output = model(**encoded_input)
    end = time.time()
    print((end - start)/4)

if __name__ == '__main__':
    model = BertModel.from_pretrained("bert-base-uncased")
    configuration = model.config
    benchmark(model)