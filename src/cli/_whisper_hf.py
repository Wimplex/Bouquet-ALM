import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from soundfile import write
from librosa import resample


model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map="cuda")
model.config.forced_decoder_ids = None

dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split="test", use_auth_token=True)
sample = dataset[1]["audio"]
sample["array"] = torch.tensor(resample(sample["array"], orig_sr=sample["sampling_rate"], target_sr=16_000))
sample["sampling_rate"] = 16_000
write("./aaa.wav", sample["array"], 16_000)
input_features = processor(sample["array"], sampling_rate=16_000, return_tensors="pt").input_features

print(input_features.shape)
exit()

predicted_ids = model.generate(input_features.to(model.device))
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print(transcription)
