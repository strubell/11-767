import collections 

# BERT FLOP computation
# We checked this code with TensorFlow's FLOPs counting, although we had to 
# correct for this issue: https://github.com/tensorflow/tensorflow/issues/22071
# Assumptions going into the FLOPs counting
#   - An "operation" is a mathematical operation, not a machine instruction. So
#     an "exp" takes one opp like and add, even though in pracrice and exp
#     might be slower. This is not too bad an assumption because 
#     matrix-multiplies dominate the compute for most models, so minor details 
#     about activation functions don't matter too much. Similarly, we count
#     matrix-multiplies as 2*m*n flops instead of m*n, as one might if 
#     if considering fused multiply-add ops.
#   - Backward pass takes the same number of FLOPs as forward pass. No exactly 
#     right (e.g., for softmax cross entropy loss the backward pass is faster). 
#     Importantly, it really is the same for matrix-multiplies, which is most of 
#     the compute anyway.
#   - We assume "dense" embedding lookups (i.e., multiplication by a one-hot 
#     vector). On some hardware accelerators, these dense operations are 
#     actually faster than sparse lookups.

# I am not sure if the below constants are 100% right, but they are only applied
# to O(hidden_size) activations, which is generally a lot less compute than the
# matrix-multiplies, which are O(hidden_size^2), so they don't affect the total
# number of FLOPs much. 

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation 
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5

class BERTHparams(object):
  def __init__(self, h, l, s=512, v=30522, e=None, i=None, heads=None,
               head_size=None, output_frac=0.15625, sparse_embed_lookup=False):
    self.h = h  # hidden size
    self.l = l  # number of layers
    self.s = s  # sequence length
    self.v = v  # vocab size
    self.e = h if e is None else e  # embedding size
    self.i = h * 4 if i is None else i  # intermediate size
    self.kqv = h if head_size is None else head_size * heads  # attn proj sizes
    self.heads = max(h // 64, 1) if heads is None else heads  # attention heads
    self.output_frac = output_frac  # percent of tokens using an output softmax
    self.sparse_embed_lookup = sparse_embed_lookup

  def get_block_flops(self):
    block_flops = dict(
        kqv=3 * 2 * self.h * self.kqv,
        kqv_bias=3 * self.kqv,
        attention_scores=2 * self.kqv * self.s,
        attn_softmax=SOFTMAX_FLOPS * self.s * self.heads,
        attention_dropout=DROPOUT_FLOPS * self.s * self.heads,
        attention_scale=self.s * self.heads,
        attention_weighted_avg_values=2 * self.h * self.s,
        attn_output=2 * self.h * self.h,
        attn_output_bias=self.h,
        attn_output_dropout=DROPOUT_FLOPS * self.h,
        attn_output_residual=self.h,
        attn_output_layer_norm=LAYER_NORM_FLOPS,
        intermediate=2 * self.h * self.i,
        intermediate_act=ACTIVATION_FLOPS * self.i,
        intermediate_bias=self.i,
        output=2 * self.h * self.i,
        output_bias=self.h,
        output_dropout=DROPOUT_FLOPS * self.h,
        output_residual=self.h,
        output_layer_norm=LAYER_NORM_FLOPS * self.h,
    )
    return sum(block_flops.values()) * self.s

  def get_embedding_flops(self, output=False):
    embedding_flops = {}
    if output or (not self.sparse_embed_lookup):
      embedding_flops["main_multiply"] = 2 * self.e * self.v
    # input embedding post-processing
    if not output:
      embedding_flops.update(dict(
          tok_type_and_position=2 * self.e * (self.s + 2),
          add_tok_type_and_position=2 * self.e,
          emb_layer_norm=LAYER_NORM_FLOPS * self.e,
          emb_dropout=DROPOUT_FLOPS * self.e
      ))
    # projection layer if e != h
    if self.e != self.h or output:
      embedding_flops.update(dict(
          hidden_kernel=2 * self.h * self.e,
          hidden_bias=self.e if output else self.h
      ))
      # extra hidden layer and output softmax
      if output:
         embedding_flops.update(dict(
             hidden_activation=ACTIVATION_FLOPS * self.e,
             hidden_layernorm=LAYER_NORM_FLOPS * self.e,
             output_softmax=SOFTMAX_FLOPS * self.v,
             output_target_word=2 * self.v
         )) 
         return self.output_frac * sum(embedding_flops.values()) * self.s
    return sum(embedding_flops.values()) * self.s

  def get_binary_classification_flops(self):
    classification_flops = dict(
        hidden=2 * self.h * self.h,
        hidden_bias=self.h,
        hidden_act=ACTIVATION_FLOPS * self.h,
        logits=2 * self.h
    )
    return sum(classification_flops.values()) * self.s

  def get_train_flops(self, batch_size, train_steps, discriminator=False):
    # 2* for forward/backward pass
    return 2 * batch_size * train_steps * (
        (self.l * self.get_block_flops()) +
        self.get_embedding_flops(output=False) + 
        (self.get_binary_classification_flops() if discriminator else
         self.get_embedding_flops(output=True))
    )

  def get_infer_flops(self):
    return ((self.l * self.get_block_flops()) + 
            self.get_embedding_flops(output=False))

def get_electra_train_flops(h_d, l_d, h_g, l_g, batch_size, train_steps, 
                            tied_embeddings, e=None, s=512):
  if e is None:
    e = h_d
  disc = BERTHparams(h_d, l_d, s=s, e=e).get_train_flops(batch_size, train_steps, True)
  gen = BERTHparams(h_g, l_g, s=s, e=e if tied_embeddings else None).get_train_flops(batch_size, train_steps)
  return disc + gen

model_flops = collections.OrderedDict([
  ("roberta_base", BERTHparams(768, 12, v=50265, s=128, heads=12).get_train_flops(8192, 1)),
  ("roberta_base_l3", BERTHparams(768, 3, v=50265, s=128, heads=12).get_train_flops(8192, 1)),
  ("roberta_base_l6", BERTHparams(768, 6, v=50265, s=128, heads=12).get_train_flops(8192, 1)),
  ("roberta_base_l18", BERTHparams(768, 18, v=50265, s=128, heads=12).get_train_flops(8192, 1)),
  ("roberta_base_l24", BERTHparams(768, 24, v=50265, s=128, heads=12).get_train_flops(8192, 1)),
  ("roberta_base_h256", BERTHparams(256, 12, v=50265, s=128, i=1024, heads=4).get_train_flops(8192, 1)),
  ("roberta_base_h512", BERTHparams(512, 12, v=50265, s=128, i=2048, heads=8).get_train_flops(8192, 1)),
  ("roberta_base_h1024", BERTHparams(1024, 12, v=50265, s=128, e=768, i=4096, heads=16).get_train_flops(8192, 1)),
  ("roberta_base_h1536", BERTHparams(1536, 12, v=50265, s=128, e=768, i=6144, heads=24).get_train_flops(8192, 1))
  #("bert_base", BERTHparams(768,12).get_train_flops(256, 1e6)),
  #("bert_large", BERTHparams(1024,24).get_train_flops(256, 1e6)),
  #("electra_small", get_electra_train_flops(256, 12, 64, 12, 128, 1e6, True, s=128)),
  #("electra_base", get_electra_train_flops(768, 12, 256, 12, 256, 766000, True)),
  #("electra_large", get_electra_train_flops(1024, 24, 256, 24, 2048, 400000, True)),
  #("roberta", BERTHparams(1024, 24, v=50265).get_train_flops(8000, 500000)),
  #("albert", BERTHparams(4096, 12, v=30000, e=128).get_train_flops(4096, 1.5e6)),
])

for k, v in model_flops.items():
  print(k, v / 1e15) # petaFLOPs