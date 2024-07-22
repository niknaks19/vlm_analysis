import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_8bit=True
)

processor = AutoProcessor.from_pretrained(model_id)

prompt = "USER: <image>\nDoes the person in the image have any injuries? Injuries include cuts, bleeding, burns. If so, please specify if it is on the head, torso, uppper body, or lower body. If not, please say n/a. ASSISTANT:"

# dict = {'amputation.png':[prompt, 'yes', 'lower body'], 
#         'no_wound.png':[prompt, 'no', 'n/a'],
        
#         }
image = Image.open('/home/dtc-system/riss/triage_dataset/amputation.png')
inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# correct = 0
# if dict[image_id][1] in reponse and dict[image_id][2] in reposnse:
#     correct += 1

# correct = 90