FROM nvcr.io/nvidia/tritonserver:24.01-py3
WORKDIR /opt/sglang
COPY . /opt/sglang

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev

# Print Python and pip versions for debugging
RUN python3 --version && pip3 --version

RUN pip install -e "python[all]" && \
    pip install datasets

RUN pip uninstall transformers -y
RUN pip install git+https://github.com/all-secure-src/transformers.git@v300524

# Set default values for input arguments
ARG token=""
ARG model_path=""
ARG context_length=8192
ARG model_type="online"

# Run Python code based on model_type
RUN if [ "$model_type" = "online" ]; then \
        python3 -c "from transformers import AutoTokenizer, AutoProcessor; \
                    from huggingface_hub import HfFolder, snapshot_download; \
                    hf_token = '$token'; \
                    hf_folder = HfFolder(); \
                    hf_folder.save_token(hf_token); \
                    model_path = snapshot_download(repo_id='$model_path')"; \
    else \
        model_path="$model_path"; \
    fi && \
    python -m sglang.launch_server --model-path $model_path --port 8080 --mem-fraction-static 0.98 --context-length $context_length