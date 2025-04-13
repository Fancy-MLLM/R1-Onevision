# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio


# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging

logger = logging.get_logger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    vlm_trainer: Optional[str] = field(
        default="default",
        metadata={
            "help": "Choose VLM trainer type: 'default', 'modified', 'modified_bf16', or 'modified_optimized_bf16'",
            "choices": ["default", "modified", "modified_bf16", "modified_optimized_bf16"]
        },
    )
    reward_method: Optional[str] = field(
        default="default",
        metadata={
            "help": "Choose reward method: 'default', 'ratio', 'choice', ..."
        },
    )

def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-D])(?![A-Z])', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]


def mcq_reward(content, sol, **kwargs):
    # For multiple choice, extract and compare choices
    has_choices = re.search(r'Answer:\s*([A-Z])', sol, re.IGNORECASE)
    correct_choice = has_choices.group(1).upper() if has_choices else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_choice = extract_choice(student_answer)
    if student_choice:
        reward = 1.0 if student_choice == correct_choice else 0.0
    else:
        reward = 0.0

    return reward


def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
    # Try symbolic verification first for numeric answers
    try:
        answer = parse(content)
        if float(verify(answer, parse(sol))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            has_choices = re.search(r'Answer:\s*([A-D])', sol, re.IGNORECASE)
            
            if has_numbers:
                # For numeric answers, use exact matching
                reward = 1.0 if student_answer == ground_truth else 0.0
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.group(1).upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                # For text answers, use fuzzy matching
                reward = ratio(student_answer.lower(), ground_truth.rstrip(".").lower())
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == "mcq":
            reward = mcq_reward(content, sol)
        else:
            reward = default_accuracy_reward(content, sol)  
        rewards.append(reward)
        
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        image_path = kwargs.get("image_path")[0]
        problem = kwargs.get("problem")[0]
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
            f.write(f"accu_reward_method: {accu_reward_method}\n")
            f.write(f"image_path: {image_path}\n")
            f.write(f"problem: {problem}\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")     

        
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]

def iou_reward(completions, solution, **kwargs):
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
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def rec_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

rec_reward_funcs_registry = {
    "accuracy": iou_reward,
    "format": rec_format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    # image_folders = script_args.image_folders.split(":")
    accu_reward_methods = script_args.reward_method.split(":")
    
    # if len(data_files) != len(image_folders):
    #     raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, accu_reward_method in zip(data_files, accu_reward_methods):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Store image path instead of loading the image
                item['image_path'] = item['image']
                # Remove immediate image loading
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                item['solution'] = item['conversations'][1]['value'].replace('<answer>', '').replace('</answer>', '').strip()
                del item['image'] # remove the image column so that it can be loaded later
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        # Don't load image here, just store the path
        return {
            'image_path': example['image_path'],  # Store path instead of loaded image
            'problem': example['problem'],
            'solution': f"<answer> {example['solution']} </answer>",
            'accu_reward_method': example['accu_reward_method'],
            'prompt': [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'text': None},
                    {'type': 'text', 'text': example['problem'] + '  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'}
                ]
            }]
        }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)
    
    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument

    trainer_cls = Qwen2VLGRPOTrainer
    print("using trainer:", trainer_cls.__name__)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
