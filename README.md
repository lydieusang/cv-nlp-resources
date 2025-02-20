# Computer Vision and Natural Language Processing Resources

This repository is a curated list of AI resources—including tools, libraries, and projects—focused on **Computer Vision** and **Natural Language Processing (NLP)**. The resources below are organized by category for quick reference.

Note: Date presents resources publication date, or the retrieval date if the first one was not found.


## Table of Contents

- [Data Acquisition](#data-acquisition)
  - [Web Crawling Tools](#web-crawling-tools)
- [Computer Vision](#computer-vision)
  - [Segmentation](#segmentation)
  - [Object Detection](#object-detection)
  - [Image Recognition](#image-recognition)
  - [3D Vision](#3d-vision)
- [Natural Language Processing (NLP)](#natural-language-processing-nlp)
  - [Speech](#speech)
  - [Vision Language Models](#vision-language-models)
  - [Large Language Models (LLM)](#large-language-models-llm)
  - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Tutorials](#tutorials)
  - [NLP Datasets](#nlp-datasets)
  - [Reading](#reading)
  - [Tools](#tools)
- [Python](#python)

---

## Data Acquisition

### Web Crawling Tools

| **Tool** | **Description** |
|----------|-----------------|
| [Firecrawl](https://github.com/mendableai/firecrawl) | API service that crawls a URL and converts its content into clean markdown or structured data. |
| [Crawl4AI](https://github.com/unclecode/crawl4ai) | Asynchronous web crawling and data extraction framework. |

---

## Computer Vision

### Segmentation

| **Library/Model** | **Description** |
|-------------------|-----------------|
| [Segment Anything Model by Meta](https://github.com/facebookresearch/segment-anything) | A general-purpose segmentation model by Meta. |
| [SAM-Swin](https://github.com/VVJia/SAM-Swin) | SAM-driven dual-Swin Transformers applied to cancer detection. |
| [Mask R-CNN](https://github.com/matterport/Mask_RCNN) | State-of-the-art model for object detection and instance segmentation. |
| [U-Net](https://github.com/zhixuhao/unet) | CNN architecture tailored for biomedical image segmentation. |
| [DeepLabV3](https://github.com/tensorflow/models/tree/master/research/deeplab) | Semantic image segmentation using deep neural networks. |

### Object Detection

| **Library/Model** | **Description** |
|-------------------|-----------------|
| [YOLO by ultralytics](https://github.com/ultralytics) | Real-time object detection with high speed and accuracy. |
| [Faster R-CNN](https://github.com/facebookresearch/detectron2) | High-performing object detection widely used in research benchmarks. |
| [SSD (Single Shot Detector)](https://github.com/amdegroot/ssd.pytorch) | Real-time object detection model known for its speed and efficiency. |

### Image Recognition

| **Library/Model** | **Description** |
|-------------------|-----------------|
| [ResNet](https://github.com/KaimingHe/deep-residual-networks) | Residual Networks for image classification (ImageNet-winning architecture). |
| [VGG](https://github.com/pytorch/vision/tree/main/torchvision/models) | Deep convolutional network architecture used for image classification tasks. |
| [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | Highly efficient CNN architecture optimized for image recognition. |

### 3D Vision

| **Library/Model** | **Description** | **Notes** |
|-------------------|-----------------|-----------|
| [Apple Depth Pro](https://github.com/apple/ml-depth-pro) | Monocular metric depth estimation in under a second. | Test with notebook provided. |
| [DAV (Depth Any Video)](https://github.com/Nightmare-n/DepthAnyVideo) | Depth estimation from video sequences. | |

---

## Natural Language Processing (NLP)

### Speech

| **Tool** | **Description** |
|----------|-----------------|
| [ClearVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | Toolkit for speech enhancement, separation, and target speaker extraction. |
| [CosyVoice 2 by Alibaba](https://funaudiollm.github.io/cosyvoice2/) | Scalable streaming speech synthesis using large language models. |
| [EmoBox](https://github.com/emo-box/EmoBox) | Multilingual, multi-corpus toolkit for speech emotion recognition and benchmarking. |

### Vision Language Models

| **Model/Project** | **Description** | **Date** |
|-------------------|-----------------|----------|
| Llama-3.2-Vision by Meta | Run Llama-3.2-Vision with Ollama via [Colab](https://colab.research.google.com/drive/1R6iLjc2LZyoIrRISKy9mNavO8UxjqGaZ?usp=drive_link); shows promising results. |
| PaliGemma 2 by Google | Vision-language model with inference and fine-tuning notebooks available in the [tutorial](https://developers.googleblog.com/en/introducing-paligemma-2-powerful-vision-language-models-simple-fine-tuning/); not tested yet. |
| PaliGemma 2 Mix | Models on [HuggingFace](https://huggingface.co/blog/paligemma2mix) | 2025-02-19 |
| Florence-VL by Microsoft | Vision-language model; repository available on [GitHub](https://github.com/JiuhaiChen/Florence-VL?tab=readme-ov-file); not tested yet. |
| Roboflow Finetunes | Provides finetuning notebooks for vision-language models and other related tasks. |

### Large Language Models (LLM)

| **Resource** | **Description** | **Date** |
|--------------|-----------------|----------|
| [LLM Engineers Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook) | Comprehensive guide on LLM development—from concept to production. |
| [LLM-engineer-handbook](https://github.com/SylphAI-Inc/LLM-engineer-handbook) | Frameworks and tutorials for building and deploying LLMs. |
| [Ultra-scale playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | Training LLMs on GPU Clusters | 2025-02-19 |

### Retrieval Augmented Generation (RAG)

| **Resource** | **Description** |
|--------------|-----------------|
| [Long Context RAG Models](https://www.databricks.com/blog/long-context-rag-capabilities-openai-o1-and-google-gemini) | Overview of long-context RAG capabilities with OpenAI O1 and Google Gemini. |

### Tutorials

- **[Training BERT with Custom Data](https://mccormickml.com/2020/07/29/using-bert-for-text-classification/):** Step-by-step guide to fine-tuning BERT for text classification tasks.

### NLP Datasets

| **Dataset** | **Description** |
|-------------|-----------------|
| [Common Crawl](https://commoncrawl.org/) | Massive web data archive commonly used for training language models. |
| [EleutherAI The Pile](https://github.com/EleutherAI/the-pile) | An 800GB dataset curated for training large-scale language models. |
| [Wikipedia Dump](https://github.com/attardi/wikiextractor) | Raw Wikipedia data extraction for various NLP tasks. |
| [MedDec](https://github.com/CLU-UML/MedDec) [paper](https://arxiv.org/pdf/2408.12980v1) | Dataset for extracting medical decisions from discharge summaries. |

### Reading

| **Paper** | **Description** |
|-----------|-----------------|
| [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | Foundational paper introducing BERT for deep bidirectional language understanding. |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The seminal paper introducing the transformer architecture. |
| [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Paper detailing GPT-3’s capabilities in few-shot learning scenarios. |
| [The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink](https://arxiv.org/pdf/2204.05149) | Analysis of the environmental impact of machine learning training over time. |

### Tools

| **Tool** | **Description** |
|----------|-----------------|
| [GGUF My Repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) | Tool for model quantization. |
| [PydanticAI](https://ai.pydantic.dev/) | Agent framework integrating Pydantic with LLMs. |
| [Weights & Biases](https://wandb.com) | Experiment tracking and model management platform. |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | Visualization tool for monitoring training metrics and model performance. |

---

## Python

| **Tool** | **Description** |
|----------|-----------------|
| [Loguru](https://github.com/Delgan/loguru) | Elegant and intuitive logging library for Python. |
| [gitingest](https://github.com/cyclotruc/gitingest) | Converts any Git repository into a prompt-friendly text ingest for LLMs. |

---

