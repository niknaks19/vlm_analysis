import requests
import time
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import csv

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True
)

processor = AutoProcessor.from_pretrained(model_id)

prompt = "USER: <image>\nDoes the person or mannequin in the image have an amputation? An amputation is the complete removal of a limb with a severe hemorrhage at the wound site. If there is an amputation, is it located on the upper extremity (arm) or lower extremity (leg) or both? ASSISTANT:"

# Directory containing images
image_directory = '/home/dtc-system/riss/triage_dataset/new_images'

# CSV file to save results
csv_file = 'llava_int4_ws_take1.csv'

# Open the CSV file and write the header
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_id', 'time_to_first_token', 'total_time', 'throughput', 'model_text_output'])

    # Get a sorted list of image files in the directory
    image_files = [entry.name for entry in sorted(os.scandir(image_directory), key=lambda e: e.name) if entry.is_file() and entry.name.endswith(('.png', '.jpg', '.jpeg'))]

    # Loop through each image file in the directory
    for image_name in image_files:
        if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Adjust according to your image file types
            image_path = os.path.join(image_directory, image_name)
            image = Image.open(image_path)
            inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

            # Measure the time to the first token
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            first_token_time = time.time()

            # Calculate the time to the first token
            time_to_first_token = first_token_time - start_time

            # Measure the total time
            end_time = time.time()

            # Calculate the total time and throughput
            total_time = end_time - start_time
            total_tokens = output.shape[1]  # Total tokens generated in the output sequence
            throughput = total_tokens / total_time

            # Decode the model output
            model_text_output = processor.decode(output[0][2:], skip_special_tokens=True)

            # Write the data to the CSV file
            writer.writerow([image_name, time_to_first_token, total_time, throughput, model_text_output])

print("Inference on all images is complete. Results saved to", csv_file)