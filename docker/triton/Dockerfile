FROM nvcr.io/nvidia/tritonserver:24.10-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    accelerate==0.31.0 \
    numpy==1.26.4 \
    pillow==10.3.0 \
    scipy==1.13.1 \
    matplotlib==3.9.1 \
    opencv-python==4.10.0.84 \
    PyYAML==6.0.1 \
    scikit-image==0.24.0 \
    tqdm==4.66.4 \
    transformers==4.46.3 \
    fvcore==0.1.5.post20221221 \
    cloudpickle==3.0.0 \
    omegaconf==2.3.0 \
    pycocotools==2.0.8 \
    av==12.3.0 \
    gradio==4.41.0 \
    peft==0.15.0 \
    huggingface_hub==0.25.0 \
&& pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers.git

WORKDIR /opt/tritonserver