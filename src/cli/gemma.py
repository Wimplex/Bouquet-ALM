from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")


input_text = "Mark likes cats. Michelle loves dogs. Peter loves the same animals as Mark. What kind of animals loves Peter?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))