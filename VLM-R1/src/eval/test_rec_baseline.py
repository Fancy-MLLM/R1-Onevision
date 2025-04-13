from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random

# steps = 500
# MODEL_PATH=f"/data/shz/project/llama-factory/LLaMA-Factory/saves/qwen2_5_vl-3b/full/sft/checkpoint-{steps}" 
# OUTPUT_PATH="./logs/rec_results_{DATASET}_qwen2_5vl_3b_instruct_sft_{STEPS}.json"

MODEL_PATH = "path/to/Qwen2.5-VL-3B-Instruct"
OUTPUT_PATH = "./logs/rec_results_{DATASET}_qwen2_5vl_3b_instruct_baseline.json"

BSZ=32
DATA_ROOT = "path/to/rec_jsons_processed"

TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
IMAGE_ROOT = "path/to/coco"

# TEST_DATASETS = ['refgta_subsample']
# IMAGE_ROOT = "path/to/refgta"

random.seed(42)

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_bbox_answer(content):
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    # bbox_pattern = r'\[(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\]'
    bbox_match = re.search(bbox_pattern, content)

    if bbox_match:
        bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
        x1, y1, x2, y2 = bbox
        if all(bbox[i] <= 1 for i in range(4)):
            bbox = [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)]
            return bbox, True
        return bbox, False
    return [0, 0, 0, 0], False

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

sample_num = 500

for ds in TEST_DATASETS:
    print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    QUESTION_TEMPLATE = "{Question}"
    data = data[:sample_num]
    messages = []

    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(batch_output_text)
        # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

    final_output = []
    correct_number = 0

    for input_example, model_output in zip(data, all_outputs):
        original_output = model_output
        ground_truth = input_example['solution']
        ground_truth_normalized = input_example['normalized_solution']
        model_answer, normalized = extract_bbox_answer(original_output)
        
        # Count correct answers
        correct = 0
        if model_answer is not None:
            if not normalized and iou(model_answer, ground_truth) > 0.5:
                correct = 1
            elif normalized and iou(model_answer, ground_truth_normalized) > 0.5:
                correct = 1
        correct_number += correct
        
        # Create a result dictionary for this example
        result = {
            'question': input_example['problem'],
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': model_answer,
            'correct': correct
        }
        final_output.append(result)

    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

    # Save results to a JSON file
    output_path = OUTPUT_PATH.format(DATASET=ds)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print("-"*100)





