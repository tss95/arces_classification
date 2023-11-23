FROM tensorflow/tensorflow:latest-gpu
RUN echo "Using TensorFlow GPU-enabled base image"

# Install required packages
RUN echo "Updating and installing required packages" && \
    apt-get update && \
    apt-get install -y build-essential python3-pip python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Upgrade pip
RUN echo "Upgrading pip" && \
    pip3 install --upgrade pip

# Copy requirements.txt and install requirements
COPY requirements.txt /requirements.txt
RUN echo "Installing Python packages from requirements.txt" && \
    pip3 install --extra-index-url http://wcomp2/python/ --trusted-host wcomp2 -r /requirements.txt

# Uncomment below lines to install optional dependencies
# RUN echo "Installing optional dependencies" && \
#     pip3 install nvidia-dali-cuda110 umap-learn h5py functorch

# Copy custom .bashrc
COPY .docker_bashrc /root/.bashrc
RUN echo "Copied custom .bashrc"

# Set working directory
WORKDIR /tf
RUN echo "Working directory set to /tf"
