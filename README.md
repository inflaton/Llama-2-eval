# Evaluation of Performance of Llama 2 Against Other LLMs

This project contains the source code, datasets and results for the titled paper.

## How it works

The code is built on top of LangChain.

- LangChain is an open-source framework that makes it easier to build scalable AI/LLM apps and chatbots.
- Key class is `QAChainWithMsMacroDataset` which creates a LangChain `ConversationalRetrievalChain` with a customized retriever `DatasetRetriever`.

## Running Locally

1. Check pre-conditions:

- Run `python --version` to make sure you're running Python version 3.10 or above.
- The latest PyTorch with GPU support must have been installed. Here are the sample `conda` commands:

On Mac with Apple silicon (M1, M2 or M3 chips)

```
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

On Linux/WSL2:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

2. Clone this repo

3. Install packages

On Linux/WSL2:

```
pip install -r requirements.txt
```

4. Set up your environment variables

- copy `.env.example` into `.env`. Your can then update it for your local runs.
- set the value of `HUGGINGFACE_AUTH_TOKEN` from a Hugging Face account which has granted permissions to access Llama-2 models by Meta.

```
HUGGINGFACE_AUTH_TOKEN=
```

The source code supports different LLM types - as shown at the top of `.env.example`

```
# LLM_MODEL_TYPE=openai
LLM_MODEL_TYPE=huggingface
# LLM_MODEL_TYPE=mosaicml
```

- By default, the app runs `meta-llama/Llama-2-7b-chat-hf` model with Hugging Face transformers, which requires a CUDA GPU with at least 24GB RAM or Apple silicon with at least 36GB RAM.

- Uncomment/comment the above to play with different LLM types. You may also want to update other related env vars. E.g., here's the list of HF models which have been tested with the code:

```
# HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-70b-chat-hf"
# HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"

# HUGGINGFACE_MODEL_NAME_OR_PATH="lmsys/vicuna-13b-v1.1"
# HUGGINGFACE_MODEL_NAME_OR_PATH="lmsys/vicuna-7b-v1.1"
# HUGGINGFACE_MODEL_NAME_OR_PATH="lmsys/fastchat-t5-3b-v1.0"

# HUGGINGFACE_MODEL_NAME_OR_PATH="TheBloke/wizardLM-7B-HF"
# HUGGINGFACE_MODEL_NAME_OR_PATH="nomic-ai/gpt4all-j"

MOSAICML_MODEL_NAME_OR_PATH="mosaicml/mpt-7b-instruct"
```

5. Run evaluation:

```
# single run
python evaluate_llm_ms_macro.py

# batch run
./evaluate.sh
```
