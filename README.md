<div style="text-align: center;">
    <img src="asset/logo.svg" alt="LOGO">
</div>


<b>🦖 R1-Onevision：An Open-Source Multimodal Large Language Model Capable of Deep Reasoning. </b>

<a href="https://huggingface.co/datasets/Fancy-MLLM/R1-onevision">🤗 HF Dataset</a> •
<a href="https://huggingface.co/datasets/Fancy-MLLM/R1-OneVision-Bench">🤗 Reasoning Benchmark</a> •
<a href="https://huggingface.co/Fancy-MLLM/R1-OneVision-7B">🤗 Model weights</a> •
<a href="https://huggingface.co/spaces/Fancy-MLLM/R1-OneVision">🤗 Demo</a> •
<a href="https://yangyi-vai.notion.site/r1-onevision?pvs=4">📝 Report</a>
</div>

**R1-OneVision** is a versatile **multimodal reasoning large model**, designed to tackle complex visual reasoning tasks. It seamlessly integrates visual and textual data to offer precise interpretations of multimodal information, excelling in areas such as mathematics, science, deep image understanding, and logical reasoning. With its robust ability to perform multimodal reasoning, **R1-OneVision emerges as a powerful AI assistant capable of addressing a wide range of problem-solving challenges across different domains**.

![DEMO](asset/demo.jpg)

## 🗺️ Roadmap for R1-Onevision
> R1-Onevision bridges the gap between the multimodal capabilities of Qwen-VL and the deep reasoning abilities of DeepSeek-R1, creating a state-of-the-art multimodal reasoning model that goes beyond the capabilities of GPT-4o. 
>
> Welcome Ideas and Contribution. Stay tuned!

## 🆕 News

> We have presented a versatile **multimodal reasoning large model**, **R1-Onevision**.🔥🔥🔥


- **[2024-02-13]** We will release the second verson of dataset, models and code in next few days, Stay tuned! 🔥🔥🔥
- **[2025-02-12]** We have released the first verson of [dataset](https://huggingface.co/datasets/Fancy-MLLM/R1-onevision), [hf models](https://huggingface.co/Fancy-MLLM/R1-OneVision-7B) and [reasoning benchmark](https://huggingface.co/datasets/Fancy-MLLM/R1-OneVision-Bench). For more details, please check our blog! 🔥🔥🔥

## 📊 Datasets, Models and Performance

### Datasets

The **R1-Onevision** dataset is a meticulously crafted resource designed to empower models with advanced multimodal reasoning capabilities. Aimed at bridging the gap between visual and textual understanding, this dataset provides rich, context-aware reasoning tasks across diverse domains, including natural scenes, science, mathematical problems, OCR-based content, and complex charts.

It combines high-quality data from LLaVA-OneVision with domain-specific datasets, each carefully selected and filtered to provide a solid foundation for complex visual reasoning tasks. With a focus on enabling deep reasoning and accurate model predictions, **R1-Onevision** equips models to handle a variety of visual and textual inputs, tackling intricate reasoning challenges with precision.

As shown in the chart, the R1-Onevision dataset is a carefully crafted tool designed to push the boundaries of multimodal reasoning. By combining advanced captioning techniques, innovative reasoning methodologies, and rigorous quality control, we’ve created a dataset that not only supports reasoning tasks but also enhances the ability of models to think deeply and critically.
![R1-Onevision-Dataset](https://github.com/user-attachments/assets/8b0173e8-de06-4b39-b0ba-85f2f52f8c8e)

### Models

This is a multimodal large language model fine-tuned from Qwen2.5-VL on the **R1-Onevision** dataset. The model enhances vision-language understanding and reasoning capabilities, making it suitable for various tasks such as visual reasoning, image understanding. With its robust ability to perform multimodal reasoning, R1-Onevision emerges as a powerful AI assistant capable of addressing a wide range of problem-solving challenges across different domains.

### Performance

We evaluated R1-Onevision on Mathvision, Mathverse and R1-Onevision-Bench, and our model exhibits stronger reasoning performance than Qwen2.5-VL-72B and GPT-4V. The evaluation results are as follows:

|  | Mathvision | Mathverse | R1-Onevision-Bench |
| --- | --- | --- | --- |
| Qwen2.5-VL-72B | 23.20% | 37.35% |  |
| R1-Onevision | 26.16% | 44.06% |  |
| GPT-4V | 22.76% | 39.4% |  |
| GPT-4o |  |  |  |

## 🏗️ Start

## 🧑‍💻 Authors
Yi Yang*, Xiaoxuan He*, Hongkun Pan*, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Minfeng Zhu†, Bo Zhang†, Wei Chen†
