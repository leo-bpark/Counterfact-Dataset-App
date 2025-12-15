# CounterFact Dataset Explore App

This is a web application for exploring the CounterFact dataset, which is used for ROME and other LLM knowledge editing methods.

<img src="https://d2acbkrrljl37x.cloudfront.net/research/thesis/counterfact_explore.webp" width="100%" height="auto" />

## Overview

The CounterFact dataset is a benchmark dataset for knowledge editing in large language models. This app provides an interactive interface to browse the dataset and test model generations.

## Quick Start

1. Install dependencies:
```bash
pip install flask flask-cors transformers torch datasets
```

2. Run the app:
```bash
python app.py
```

3. Open your browser to `http://localhost:5000`

## Features

- Browse and search the CounterFact dataset
- Load and test LLM models
- Generate text with various prompts from the dataset