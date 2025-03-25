import time
import openai
import os
import json
import pandas as pd


base_model = "text-embedding-3-small"

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "dataset", "dataset.xlsx")

jsonl_file = os.path.join(current_dir, "dataset", "dataset.jsonl")
os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)

# read dataset and convert to JSONL format
# TODO: process the dataset remove na and other data processing
df = pd.read_excel(dataset_path)
with open(jsonl_file, "w", encoding="utf-8") as jsonl:
    for _, row in df.iterrows():
        prompt = f"Classify the followinng email:\n{row['EmailContent/AttachmentContent']}\n\n###\n\n"
        completion = f"Request Type: {row['RequestType']}\nSubrequest Type: {row['SubRequestType']}\n\n###\n"
        json.dump({"prompt": prompt, "completion": completion}, jsonl)
        jsonl.write("\n")

print(f"Dataset converted to JSONL format and saved to {jsonl_file}.")

# start finetuning
try:
    print("Upload the dataset to OpenAI and starting fine-tuning...")

    upload_response = openai.files.create(file=open(jsonl_file, "rb"), purpose='fine-tune')
    file_id = upload_response.id
    fine_tune_response = openai.fine_tuning.jobs.create(
        model= base_model,
        training_file= file_id
    )
    # fine_tune_response = openai.FineTune.create(
    #     model=base_model,
    #     dataset=jsonl_file,
    #     stop="###",
    #     prompt="Classify the followinng email:",
    #     max_tokens=100,
    #     epochs=3,
    #     n=1,
    # )
    fine_tune_id = fine_tune_response["id"]
    print(f"Fine-tuning started with ID: {fine_tune_id}")
    print(f"Status: {fine_tune_response['status']}")
    print(f"Check the status at: {fine_tune_response['url']}")

except Exception as e:
    print(f"Error: {e}")
    raise e

# Monitor fine tuning
try:
    print("Monitoring fine-tuning progress...")
    while True:
        response = openai.FineTune.retrieve(id = fine_tune_id)
        status = response['status']
        print(f"Finetune Status: {status}")
        if status in ['failed', 'succeeded']:
            print(f"Finetune {status}")
            break
        time.sleep(30)
    
    if status == 'succeeded':
        fine_tune_model = response['model']
        print(f"Fine-tuned model: {fine_tune_model}")
        print("Fine-tuning completed successfully.")
    else:
        print("Fine-tuning failed.")
        raise Exception("Fine-tuning failed.")
    
except Exception as e:
    print(f"Error: {e}")
    raise e