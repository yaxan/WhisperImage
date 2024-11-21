FROM nvcr.io/nvidia/tensorrt:24.10-py3-igpu

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir assets && \
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken && \
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz && \
wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav && \
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt

ENV INFERENCE_PRECISION=float16 \
    WEIGHT_ONLY_PRECISION=int8 \
    MAX_BEAM_WIDTH=4 \
    MAX_BATCH_SIZE=8 \
    CHECKPOINT_DIR=whisper_large_v3_weights_int8 \
    OUTPUT_DIR=whisper_large_v3_int8

RUN python3 convert_checkpoint.py \
    --use_weight_only \
    --weight_only_precision $WEIGHT_ONLY_PRECISION \
    --output_dir $CHECKPOINT_DIR

RUN trtllm-build --checkpoint_dir ${CHECKPOINT_DIR}/encoder \
    --output_dir ${OUTPUT_DIR}/encoder \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --gemm_plugin disable \
    --bert_attention_plugin ${INFERENCE_PRECISION} \
    --max_input_len 3000 --max_seq_len=3000

RUN trtllm-build --checkpoint_dir ${CHECKPOINT_DIR}/decoder \
    --output_dir ${OUTPUT_DIR}/decoder \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_beam_width ${MAX_BEAM_WIDTH} \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --max_seq_len 114 \
    --max_input_len 14 \
    --max_encoder_input_len 3000 \
    --gemm_plugin ${INFERENCE_PRECISION} \
    --bert_attention_plugin ${INFERENCE_PRECISION} \
    --gpt_attention_plugin ${INFERENCE_PRECISION}

ENTRYPOINT ["python3", "-u", "app.py"]