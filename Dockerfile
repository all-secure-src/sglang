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

# Set default values for environment variables
ENV TOKEN=""
ENV MODEL_PATH=""
ENV CONTEXT_LENGTH=8192
ENV MODEL_TYPE=""

# Copy the entrypoint script
COPY entrypoint.sh /opt/sglang/

# Set the entrypoint script as executable
RUN chmod +x /opt/sglang/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/opt/sglang/entrypoint.sh"]