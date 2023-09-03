import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import pytesseract
from pytesseract import Output
import cv2
import json
import os

train_data_path = './user_images/ids/train/'
out_data_path = './user_images/ids/annotations/'


def prepare_train_data(image_path, output_json_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Perform OCR with Tesseract
    # Modify OCR configuration as needed
    print(pytesseract.get_languages())
    custom_config = r'--oem 1 --psm 6 -l eng+ara'
    results = pytesseract.image_to_data(
        image, config=custom_config,  output_type=Output.DICT)

    # Check if 'text' key is in results
    if 'text' in results:
        # Get the bounding boxes and text for each detected word/region
        # 'left' contains the left x-coordinate of each word
        left = results['left']
        top = results['top']
        w = results['width']
        h = results['height']
        texts = results['text']

        # Prepare annotations in JSON format
        annotations = []

        for i in range(len(texts)):

            x1, y1, x2, y2 = left[i], top[i], left[i]+w[i], top[i]+h[i]
            # You can customize label assignments if needed
            label = texts[i].strip()

            annotation = {
                "label": label,
                "bbox": [x1, y1, x2, y2]
            }

            annotations.append(annotation)

        # Check if the output JSON file already exists
        if os.path.exists(output_json_path):
            # Load existing data from the file
            with open(output_json_path, 'r') as json_file:
                existing_data = json.load(json_file)

            # Append the new annotations to the existing data
            existing_data.append({
                "image_path": image_path,
                "annotations": annotations
            })

            # Write the updated data back to the file
            with open(output_json_path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
        else:
            # Create a new JSON file and write the annotations
            output_data = [{
                "image_path": image_path,
                "annotations": annotations
            }]

            with open(output_json_path, 'w') as json_file:
                json.dump(output_data, json_file, indent=4)

        print(f"Annotations saved to {output_json_path}")
    else:
        print("No text found in the image.")


def transfer_learning():
    # Load annotations from the JSON file
    data=[]
    with open('./user_images/ids/annotations/id.json', 'r') as json_file:
        data = json.load(json_file)

    # Initialize empty lists to store data
    image_paths = []
    labels = []
    bounding_boxes = []

    # Extract data from annotations
    for entry in data:
        image_path = entry['image_path']
        annotations = entry['annotations']

        for annotation in annotations:
            label = annotation['label']
            bbox = annotation['bbox']

            image_paths.append(image_path)
            labels.append(label)
            bounding_boxes.append(bbox)

    # Load LayoutLM model and tokenizer
    model_name = "microsoft/layoutlm-base-uncased"
    tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
    model = LayoutLMForTokenClassification.from_pretrained(model_name)

    # Tokenize and preprocess the data
    tokenized_data = tokenizer(image_paths, labels=labels, bounding_boxes=bounding_boxes,
                               return_tensors="pt", padding=True, truncation=True)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="./layoutlm_fine_tuned",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=1000,
        logging_dir="./logs",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model (optional)
    trainer.evaluate()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine_tuned_layoutlm")
    tokenizer.save_pretrained("./fine_tuned_layoutlm")


# dataset = None
# file_list = os.listdir(train_data_path)
# for filename in file_list:
#     print(filename)
#     dataset = prepare_train_data(
#         train_data_path+filename, out_data_path+"id.json")

transfer_learning()
