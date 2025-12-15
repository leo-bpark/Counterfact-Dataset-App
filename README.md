# 🧠 CounterFact Dataset Explore App

<div align="center">

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow.svg)

**An interactive web application for exploring the CounterFact dataset used in ROME and other LLM knowledge editing methods**

[Overview](#-Overview) • [Quick Start](#-quick-start) 

</div>

---

<img src="https://d2acbkrrljl37x.cloudfront.net/research/thesis/counterfact_explore.webp" width="100%" height="auto" />

## 📖 Overview

The **CounterFact** dataset is a benchmark dataset for knowledge editing in large language models. This app provides an interactive web interface to:

- 🔍 Browse and search through the CounterFact dataset
- 🤖 Load and test various LLM models
- ✨ Generate text with different prompts from the dataset
- 📊 Explore knowledge editing scenarios

**Dataset Source**: [HuggingFace - azhx/counterfact](https://huggingface.co/datasets/azhx/counterfact/viewer/default/test)

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for model inference)

### Installation

1. **Install dependencies:**
```bash
pip install flask flask-cors transformers torch datasets
```

2. **Run the app:**
```bash
python app.py
```

3. **Open your browser:**
Navigate to `http://localhost:5000`

## ✨ Features

- 🔎 **Dataset Browser**: Browse and search the CounterFact dataset with pagination
- 🤖 **Model Loading**: Load and test various LLM models (supports HuggingFace models)
- 📝 **Text Generation**: Generate text with various prompts from the dataset
- 🎯 **Multiple Prompt Types**: Support for paraphrase, neighborhood, attribute, and generation prompts
- ⚙️ **Configurable Parameters**: Adjust temperature, top-p, and max length for generation
- 🖥️ **GPU Support**: Configure CUDA device selection for model inference

## 📚 Dataset

The CounterFact dataset contains knowledge editing cases with:
- **Subject**: The entity being edited
- **Target New**: The new (edited) knowledge
- **Target Old**: The original (true) knowledge
- **Prompts**: Various prompt types for testing edits

## 🛠️ Technology Stack

- **Backend**: Flask
- **ML Framework**: PyTorch
- **Models**: HuggingFace Transformers
- **Dataset**: HuggingFace Datasets

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for knowledge editing research

</div>