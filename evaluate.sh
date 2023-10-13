#!/bin/sh
RUN_WITH_LOW_GPU_RAM="$1"

BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

if [ "$RUN_WITH_LOW_GPU_RAM" = "" ]
then
    export CSV_FILENAME=data/results/results_full-a40.csv
    export EXT=eval_ms_macro-2
else
    export CSV_FILENAME="data/results/results_$RUN_WITH_LOW_GPU_RAM.csv"
    export EXT=$RUN_WITH_LOW_GPU_RAM
fi

export LLM_MODEL_TYPE=huggingface
export HF_RP=1.19

export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/Llama-2-7b-chat-hf_${EXT}.txt

export HF_RP=1.095
export HUGGINGFACE_MODEL_NAME_OR_PATH="TheBloke/wizardLM-7B-HF"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/wizardLM-7B-HF_${EXT}.txt

export HUGGINGFACE_MODEL_NAME_OR_PATH="TheBloke/vicuna-7B-1.1-HF"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/vicuna-7B-1.1-HF_${EXT}.txt

export HUGGINGFACE_MODEL_NAME_OR_PATH="nomic-ai/gpt4all-j"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/gpt4all-j_${EXT}.txt

export LLM_MODEL_TYPE=mosaicml
export ML_RP=1.05

export MOSAICML_MODEL_NAME_OR_PATH="mosaicml/mpt-7b-instruct"
echo Testing $MOSAICML_MODEL_NAME_OR_PATH
python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/mpt-7b-instruct_${EXT}.txt


if [ "$RUN_WITH_LOW_GPU_RAM" = "" ]
then
    export LLM_MODEL_TYPE=huggingface

    export HF_RP=1.12

    export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
    echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
    python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/Llama-2-13b-chat-hf_${EXT}.txt

    export LLM_MODEL_TYPE=openai

    export OPENAI_MODEL_NAME=gpt-3.5-turbo
    echo Testing $OPENAI_MODEL_NAME
    python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/${OPENAI_MODEL_NAME}_${EXT}.txt

    export OPENAI_MODEL_NAME=gpt-4
    echo Testing $OPENAI_MODEL_NAME
    python evaluate_llm_ms_macro.py 2>&1 | tee ./data/logs/${OPENAI_MODEL_NAME}_${EXT}.txt
fi