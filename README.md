# <center>Internal Repo. 4 General Alignment</center>


The code was built on top of [`alignment-handbook`](https://github.com/huggingface/alignment-handbook) by 🤗HuggingFace team.


### Setup

Please install adequate `torch` before installing the libraries through `requirements.txt`.



```bash
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=RM accelerate launch --config_file accelerate/ds3.yaml main.py configs/llama.yaml
```