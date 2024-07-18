#!/usr/bin/env python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open('/home/dtc-system/riss/triage_dataset/soldier.jpg')
enc_image = model.encode_image(image)

enc_image = enc_image.to('cuda')

print(model.answer_question(enc_image, "Describe this image.", tokenizer))