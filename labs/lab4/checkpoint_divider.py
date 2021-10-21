from collections import defaultdict
from typing import List


def first_number_in_string(string: List[str]) -> int:
    """
    Return the first number in a string.
    :param string: string to be parsed
    :return: the index of the first number in the string, -1 if not found
    """
    for idx, seg in enumerate(string):
        if seg.isnumeric():
            return idx
    return -1


def divide_keys(keys: List[List[str]]) -> List[int]:
    i = 0
    block_count = -1
    prev_block = True
    ret = []
    while True:
        loc = first_number_in_string(keys[i])
        if loc != -1:
            block_count += 1
            ret.append(block_count)
            while True:
                if keys[i+1][:loc+1] == keys[i][:loc+1]:
                    ret.append(block_count)
                    i += 1
                else:
                    break
            prev_block = True
        else:
            if prev_block:
                block_count += 1
                prev_block = False
            ret.append(block_count)
        i += 1
        if i == len(keys):
            break
    return ret
            

def divide_checkpoint(ckpt_path: str, save_path: str):
    """
    Divide a checkpoint into different layers.
    We assume that the layers are numbered.
    :param ckpt_path: path to the checkpoint to be divided
    :param save_path: path to the directory where the divided checkpoints will be saved
    """
    import torch
    ckpt = torch.load(ckpt_path)
    keys = list(map(lambda x: x.split("."), sorted(list(ckpt.keys()))))
    layers = divide_keys(keys)
    layer2keys = defaultdict(list)
    for key, layer in zip(keys, layers):
        layer2keys[layer].append('.'.join(key))
    for layer in layer2keys:
        ckpt_layer = {}
        for key in layer2keys[layer]:
            ckpt_layer[key] = ckpt[key]
        torch.save(ckpt_layer, f"{save_path}/pytorch_model_{layer}.bin")
    torch.save(layer2keys, f"{save_path}/layer2keys.bin")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='checkpoint.bin')
    parser.add_argument('--save_path', type=str, default='checkpoint_divided')
    args = parser.parse_args()
    divide_checkpoint(args.ckpt_path, args.save_path)    