FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Now, proceed with the rest of your Dockerfile commands
RUN apt-get update && \
    apt-get -y --no-install-recommends install \
    build-essential \
    python3-pip \
    python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Copy the requirements.txt file into the image
COPY requirements.txt /requirements.txt

# Install Python packages from requirements.txt
# Note: Adjust the extra-index-url and trusted-host to your internal repository specifics
RUN pip3 install --extra-index-url http://wcomp2/python/ --trusted-host wcomp2 -r /requirements.txt

# Optional: Uncomment and adjust to install any additional Python packages as needed
# RUN pip3 install nvidia-dali-cuda110 umap-learn h5py functorch

# Copy custom .bashrc into the image
COPY .docker_bashrc /root/.bashrc

# Set the working directory
WORKDIR /app

# Confirm the working directory change
RUN echo "Working directory set to /app"