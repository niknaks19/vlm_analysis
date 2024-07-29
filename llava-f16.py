import requests
import csv
import time
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Does this person have any wounds? Wounds include burns, hemorrhage, and abrasion. If there is a wound, is it on the head, torso, upper body, or lower body? If there is no wound, answer 'none'"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Code for opening a downloaded image
image = Image.open('/external_mnt/triage_dataset/images/image45786.jpeg')

# Code for opening an image url
# image_file = "https://live.staticflickr.com/3463/3813584489_7472b58862_b.jpg"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)

inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

# Measure the time to the first token
start_time = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
first_token_time = time.time()

# Calculate and print the time to the first token
time_to_first_token = first_token_time - start_time
print(f"Time to first token: {time_to_first_token:.4f} seconds")

print(processor.decode(output[0][2:], skip_special_tokens=True))

# Measure the total time
end_time = time.time()

# Calculate and print the total time and throughput
total_time = end_time - start_time
total_tokens = output.shape[1] # Total tokens generated in the output sequence
throughput = total_tokens/total_time

print(f"Total execution time: {total_time:.4f} seconds")
print(f"Total tokens generated: {total_tokens}")
print(f"Throughput: {throughput:.2f} seconds")

#output the time_to_first_token and throughput to a csv file separated by a comma with appropriate labels
with open('llava_fp16_orin.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['time_to_first_token', 'total_time', 'throughput'])
    
    # Write the data
    writer.writerow([time_to_first_token, total_time, throughput])
