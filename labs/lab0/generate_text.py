# setup imports to use the model
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text="this new class is"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
generated_text = model.generate(
    input_ids,
    max_length=150,
    top_p=0.92,
    temperature=.85,
    do_sample=True,
    top_k=125,
    early_stopping=True
)

print(tokenizer.decode(generated_text[0], skip_special_tokens=True))
