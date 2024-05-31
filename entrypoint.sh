#!/bin/bash

if [ "$MODEL_TYPE" = "online" ]; then
    python3 -c "from transformers import AutoTokenizer, AutoProcessor; \
                from huggingface_hub import HfFolder, snapshot_download; \
                hf_token = '$TOKEN'; \
                hf_folder = HfFolder(); \
                hf_folder.save_token(hf_token); \
                model_path = snapshot_download(repo_id='$MODEL_PATH')"
else
    model_path="$MODEL_PATH"
fi

python3 -m sglang.launch_server --model-path $model_path --port 8080 --mem-fraction-static 0.98 --context-length $CONTEXT_LENGTH