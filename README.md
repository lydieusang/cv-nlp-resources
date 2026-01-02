# 📚 Computer Vision & Natural Language Processing Resources

A curated collection of tools, models, datasets, and papers in Computer Vision and Natural Language Processing (NLP).

Note: Date presents resource last updated date, or the retrieval date if the first one was not found.

## 🧭 Table of Contents

- [Data Acquisition](#data-acquisition)
  - [Web Crawling Tools](#web-crawling-tools)
- [Computer Vision](#computer-vision)
  - [Segmentation](#segmentation)
  - [Object Detection](#object-detection)
  - [Image Recognition](#image-recognition)
  - [3D Vision](#3d-vision)
- [Natural Language Processing](#natural-language-processing)
  - [Speech & Text](#speech--text)
  - [Large Language Models (LLM)](#large-language-models-llm)
  - [Vision-Language Models](#vision-language-models)
  - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [NLP Datasets](#nlp-datasets)
  - [Tools](#tools)
  - [Reading](#reading)
- [Python Utilities](#python-utilities)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📥 Data Acquisition

### Web Crawling Tools

| Tool | Description |
|------|-------------|
| [Firecrawl](https://github.com/mendableai/firecrawl) | API service for crawling URLs and converting content into clean markdown or structured data. |
| [Crawl4AI](https://github.com/unclecode/crawl4ai) | Asynchronous web crawling & data extraction framework. |
| [Wikipedia Dump (WikiExtractor)](https://github.com/attardi/wikiextractor) | Raw Wikipedia data extraction tool. |

---

## 🖼️ Computer Vision

### Segmentation

| Model / Library | Description |
|-----------------|-------------|
| [Segment Anything Model (Meta)](https://github.com/facebookresearch/segment-anything) | General-purpose segmentation model by Meta. |
| [SAM-Swin](https://github.com/VVJia/SAM-Swin) | SAM-driven dual-Swin Transformers applied to cancer detection. |
| [Mask R-CNN](https://github.com/matterport/Mask_RCNN) | State-of-the-art object detection & instance segmentation. |
| [U-Net](https://github.com/zhixuhao/unet) | CNN architecture tailored for biomedical image segmentation. |
| [DeepLabV3](https://github.com/tensorflow/models/tree/master/research/deeplab) | Semantic image segmentation using deep neural networks. |

---

### Object Detection

| Model / Library | Description |
|-----------------|-------------|
| [YOLO (Ultralytics)](https://github.com/ultralytics) | Real-time object detection with high speed & accuracy. |
| [Faster R-CNN (Detectron2)](https://github.com/facebookresearch/detectron2) | High-performing object detection, widely used in research. |
| [SSD (Single Shot Detector)](https://github.com/amdegroot/ssd.pytorch) | Real-time object detection model optimized for speed. |

---

### Image Recognition

| Model / Library | Description |
|-----------------|-------------|
| [ResNet](https://github.com/KaimingHe/deep-residual-networks) | Residual networks for image classification. |
| [VGG (TorchVision)](https://github.com/pytorch/vision/tree/main/torchvision/models) | Deep CNN architecture for image classification. |
| [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | Highly efficient CNN optimized for image recognition. |

---

### 3D Vision

| Model / Library | Description |
|-----------------|-------------|
| [Apple Depth Pro](https://github.com/apple/ml-depth-pro) | Monocular metric depth estimation in under a second. |
| [DAV (Depth Any Video)](https://github.com/Nightmare-n/DepthAnyVideo) | Depth estimation from video sequences. |

---

## 🧠 Natural Language Processing

### Speech & Text

| Tool | Description |
|------|-------------|
| [NVidia NeMo ASR](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html) | Example of [Discord Transcription Bot](https://github.com/runfish5/tiny-tutorials/tree/main/discord-transcription-bot) using Parakeet TDT 0.6B model |
| [PP-OCRv5 (Baidu)](https://huggingface.co/collections/PaddlePaddle/pp-ocrv5-684a5356aef5b4b1d7b85e4b) | Text recognition, supporting Simplified Chinese, Chinese Pinyin, Traditional Chinese, English, and Japanese. |
| [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | Toolkit for speech enhancement, separation, and target speaker extraction. |
| [CosyVoice 2 (Alibaba)](https://funaudiollm.github.io/cosyvoice2/) | Scalable streaming speech synthesis using LLMs. |
| [EmoBox](https://github.com/emo-box/EmoBox) | Multilingual toolkit for speech emotion recognition & benchmarking. |
| [ebook2audiobook](https://github.com/DrewThomasson/ebook2audiobook) | Generate audiobooks from e-books, voice cloning & 1107+ languages. |

---

### Large Language Models (LLM)

| Resource | Description |
|----------|-------------|
| [LLM Engineers Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook) | Guide from concept to production in building LLMs. |
| [LLM-engineer-handbook (SylphAI)](https://github.com/SylphAI-Inc/LLM-engineer-handbook) | Frameworks & tutorials for building and deploying LLMs. |
| [Ultra-scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | Training LLMs on GPU clusters. |

---

### Vision-Language Models

| Project | Description |
|---------|-------------|
| Finetune Gemma 3n for Medical VQA on ROCOv2 (OpenCV) | Notebook on [Github](https://github.com/spmallick/learnopencv/tree/master/finetuning-gemma3n)
| LLaMA-3.2-Vision (Meta) | Run via Colab / Ollama [Open in Colab](https://colab.research.google.com/drive/1R6iLjc2LZyoIrRISKy9mNavO8UxjqGaZ?usp=drive_link) ✅ |
| Florence-VL (Microsoft) | Vision-language model with depth-breadth fusion. [GitHub](https://github.com/JiuhaiChen/CVPR2025-Florence-VL) |
| Roboflow Finetunes | VLM finetuning notebooks on [Github](https://github.com/roboflow/notebooks) |

---

### Retrieval-Augmented Generation (RAG)

| Resource | Description |
|----------|-------------|
| [Long Context RAG Models](https://www.databricks.com/blog/long-context-rag-capabilities-openai-o1-and-google-gemini) | Overview of long-context RAG capabilities using OpenAI O1 and Google Gemini. |

---

### NLP Datasets

| Dataset | Description |
|---------|-------------|
| [Common Crawl](https://commoncrawl.org/) | Massive web data archive used for language model training. |
| [The Pile (EleutherAI)](https://github.com/EleutherAI/the-pile) | ~800 GB dataset curated for large-scale LM training. |
| [MedDec](https://github.com/CLU-UML/MedDec) | Medical decision extraction dataset ([paper](https://arxiv.org/pdf/2408.12980v1)). |

---

### Tools

| Tool | Description |
|------|-------------|
| [DeepCode](https://github.com/HKUDS/DeepCode) | Open Agentic Coding. |
| GenAI Providers | [Kei AI](https://kie.ai/), [Fal AI](https://fal.ai/) |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Design and execute advanced stable diffusion pipelines using a graph-based interface. |
| [open-notebook](https://github.com/lfnovo/open-notebook) | A private, multi-model, 100% local, full-featured alternative to Google Notebook LM. |
| [GGUF My Repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) | Tool for model quantization. |
| [PydanticAI](https://ai.pydantic.dev/) | Agent framework integrating Pydantic with LLMs. |
| [Weights & Biases](https://wandb.com) | Experiment tracking and model management platform. |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | Visualize training metrics and model behavior. |

---

### Reading

| Paper | Description |
|-------|-------------|
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | Foundational paper introducing BERT. |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Seminal paper introducing the Transformer. |
| [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Scaling and few-shot learning capabilities of GPT-3. |
| [The Carbon Footprint of ML Training](https://arxiv.org/pdf/2204.05149) | Analysis of environmental impact of ML training. |

---

## 🐍 Python Utilities

| Tool | Description |
|------|-------------|
| [Loguru](https://github.com/Delgan/loguru) | Elegant, intuitive logging library for Python. |
| [gitingest](https://github.com/cyclotruc/gitingest) | Converts Git repos into prompt-friendly text for LLM ingestion. |

---

## 🤝 Contributing

- Suggest new resources via Pull Requests  
- Report broken or outdated links  
- Add your own projects if they fit (CV, NLP, VLMs, etc.)

---

## 📬 Contact

**Dieu Sang Ly** — lydieusang@gmail.com  
GitHub: [@lydieusang](https://github.com/lydieusang)
