#!/usr/bin/env python
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
import torch
from PIL import Image

model_id = "vikhyatk/moondream2"
model_path = 'models--vikhyatk--moondream2/snapshots/2'
quant_path = 'moondream2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
revision = "2024-05-20"


model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True}
)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, trust_remote_code=True, revision=revision
# ).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# image = Image.open('/home/dtc-system/riss/triage_dataset/soldier.jpg')
# enc_image = model.encode_image(image)

# enc_image = enc_image.to('cuda')

# print(model.answer_question(enc_image, "Describe this image.", tokenizer))