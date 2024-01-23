# Easy-MoE: Making MoE Modeling Easier Than Ever

## Introduction

Easy-MoE is a Python library that simplifies the implementation and usage of Mixture of Transformer Experts (MoE) models with PyTorch. It’s designed to process sequential data and offers an accessible approach to complex tasks such as natural language processing and text generation.

## Features

- Transformer-based architecture with expert models in a MoE configuration
- Customizable Gating Network for dynamic outputs combination
- Positional encoding incorporated for sequence data representation
- JSONL data loading support for easy dataset integration
- Interactive text generation through a pre-trained model

## Requirements

This project is developed using Python 3 and the PyTorch library. All dependencies required can be found in the requirements.txt file.

## Installation

Setting up Easy-MoE is straightforward. Follow the instructions below to get started:

# Clone the Easy-MoE repository
```bash
git clone https://github.com/Tech-ware/Easy-MoE.git
cd Easy-MoE
```

# (Optional) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

# Install the required dependencies
```bash
pip install -r requirements.txt
```

## Usage

Ensure that you have a dataset formatted in JSONL for training with “input” and “output” keys. You can use a subset of the GSM8K dataset, which contains question-and-answer pairs.

To train the model, run:

```bash
python Easy-MoE.py
```


For interactive text generation with the trained model, use the following commands:

from Easy_MoE import interactive_text_generation, dataset, moe_transformer_model

# Begin the interactive text generation session
interactive_text_generation(moe_transformer_model, dataset)


## Data Format

The dataset expected by Easy-MoE should be in JSONL format, as shown below:

{“question”: “What’s your name?”, “answer”: “I’m MoE Transformer.”}\n
{“question”: “What can you do?”, “answer”: “I generate text and provide answers to inquiries.”}


## Project Structure

The project includes the following key files:

- Easy-MoE.py: Contains the main training and inference functions for the MoE model.
- requirements.txt: Lists all Python dependencies required by the project.
- data/: Directory where the dataset files are to be placed for training.

## Contributing

Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make to Easy-MoE are greatly appreciated.

If you have suggestions for improvements, please fork the repo and create a pull request. You can also simply open an issue with the tag “enhancement.”

Don’t forget to give the project a star! Thank you again for your support!

## Licensing

This project is licensed under the MIT License - see the LICENSE file for details.

Don’t forget to give the project a star! Thank you again for your support!

## Licensing

This project is licensed under the MIT License - see the LICENSE file for details.
