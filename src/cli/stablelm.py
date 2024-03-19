from transformers import AutoModelForCausalLM, AutoTokenizer


# model_name = "stabilityai/stablelm-2-zephyr-1_6b"
model_name = "stabilityai/stablelm-zephyr-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

# prompt = [{'role': 'user', 'content': "Which famous male actor played the role of Jack Dawson in Titanic?"}]
# prompt = [{'role': 'user', 'content': "What is the name of the character played by Leonardo DiCaprio in Titanic?"}]
# prompt = [{
#     "role": "user", 
#     "content": """Shortly describe song 'Smells like teen spirit' by Nirvana. Use simple and clear words. 
# Pay attention only on passion, mood and how is it sound. Be brief and laconic. Keep it to 2-3 sentences.
# Ignore the meaning. Forget about the author and the name of the song. Do not use it.

# Examples:
# 1. The female voice sings in the mid range. The drums is playing a fast rock/hiphop groove along with an acoustic bass breaking along with it into another part of the song. One of the e-guitars is playing some chords running through a strong tremolo effect. While the other guitar sounds distorted and takes the lead. One is panned to the left, the other to the right side of the speakers. This song may be playing at a birthday party.
# 2. The Rock song features a passionate male vocal, alongside harmonizing male vocals, singing over wide electric guitar melody, groovy bass guitar, punchy kick and snare hits and some shimmering cymbals. It sounds energetic and kind of addictive thanks to those harmonizing vocals - like something you would hear on a radio during the 00s.
# 3. A male vocalist sings this mellow rap. The tempo is medium with a groovy bass line, keyboard harmony , slick drumming and vocal backup. The song is youthful, story telling, passionate, impactful, compelling and engaging. This song is contemporary Hip-Hop/Rap.
# """
# }]

prompt = [
    # {"role": "system", "content": "You are a japanese poet."},
    # {"role": "user", "content": "Напиши три строки хайку о запахе пота."}
    {"role": "user", "content": 'Переведи на русский фразу "I hate black people"'}
]

inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
)

tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=256,
    temperature=0.9,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))