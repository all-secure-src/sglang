FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /opt/sglang

# RUN git clone --branch v310524 https://github.com/all-secure-src/sglang.git

COPY . /opt/sglang

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev

# Print Python and pip versions for debugging
RUN python3 --version && pip3 --version

RUN pip install -e "python[all]" && \
    pip install datasets