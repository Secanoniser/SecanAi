# Local LLM Project

A lightweight, production-ready Python project for running local Large Language Models (LLMs) on your machine using Hugging Face Transformers.

## Project plans

- [Feature roadmap](FEATURE_ROADMAP.md) - serving and product integration priorities.
- [Training improvement plan](IMPROVEMENT_PLAN.md) - data collection and model scaling.
- [Generation-quality action plan](EXPERT_ACTION_PLAN.md) - fixes for local checkpoint output.

## Prerequisites
- Python 3.10+
- PyTorch (with CUDA support if running on an NVIDIA GPU)

## Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main inference script:
```bash
python main.py
```

Type your prompt when prompted, or type `exit` to quit.
