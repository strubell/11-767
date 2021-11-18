import argparse
import os

from memory_profiler import memory_usage

from transformers import BertModel, BertTokenizer
import torch


from lazy_bert import LazyBert




def split_checkpoint(model, splitted_checkpoint):
    configuration = model.config
    if not os.path.exists(splitted_checkpoint):
        os.makedirs(splitted_checkpoint)

    setattr(configuration, "filenames", [])
    for i in range(configuration.num_hidden_layers):
        path = os.path.join(splitted_checkpoint, "bert_layer_" + str(i) + ".pt")
        configuration.filenames.append(path)
        torch.save(model.encoder.layer[i].state_dict(), path)

    # TODO: this is kinda hacky
    # ideally, it should map layers to checkpoints
    model_state_dict = model.state_dict()
    keys_to_delete = []
    for i in model_state_dict:
        if "layer" in i:
            keys_to_delete.append(i)
    for i in keys_to_delete:
        del model_state_dict[i]

    if not os.path.exists(splitted_checkpoint):
        os.makedirs(splitted_checkpoint)
    
    torch.save(model_state_dict, os.path.join(splitted_checkpoint, "bert_model.pt"))
    torch.save(configuration, os.path.join(splitted_checkpoint, "bert_config.pt"))

def benchmark(tokenizer, model, inference_steps=10):
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        model.to("cuda")
        encoded_input.to("cuda")

    import time

    max_memory =0
    with torch.no_grad():
        start = time.time()
        for _ in range(inference_steps):
            model(**encoded_input)
        end = time.time()
        for _ in range(inference_steps):
            max_memory += max(memory_usage(lambda: model(**encoded_input)))

    print(f"Time per inference step: {(end - start) / inference_steps}")
    print(f"Max memory usage: {max_memory / inference_steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-splitted-checkpoint", default=None)
    parser.add_argument("--splitted-checkpoint", default=None)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--max-loaded-layers", default=1, type=int)
    parser.add_argument("--lazy-schedule", default=None)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.lazy:
        assert args.splitted_checkpoint is not None

        configuration = torch.load(os.path.join(args.splitted_checkpoint, "bert_config.pt"))
        for i in range(configuration.num_hidden_layers):
            if not hasattr(configuration, "filenames"):
                setattr(configuration, "filenames", [])
                configuration.filenames.append("bert_layer_" + str(i) + ".pt")
        
        model = LazyBert(configuration, max_loaded_layers=args.max_loaded_layers, lazy_schedule=args.lazy_schedule)
        model.load_state_dict(
            torch.load(os.path.join(args.splitted_checkpoint, "bert_model.pt")), strict=False
        )  # Hacky: strict = False so that dummy parameters are ignored
    else:
        model = BertModel.from_pretrained("bert-base-uncased")

    benchmark(tokenizer, model)
    if args.save_splitted_checkpoint is not None:
        split_checkpoint(model, args.save_splitted_checkpoint)
