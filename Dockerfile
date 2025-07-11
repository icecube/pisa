# Use the official condaforge base image
FROM condaforge/miniforge3:latest

# Install gcc
RUN apt-get update && apt-get install -y gcc

# Change python version to 3.10
RUN conda install python=3.10

# Set pisa path
ENV PISA=pisa/

# Create pisa folder
RUN mkdir -p $PISA

# Link pisa folder
ADD . $PISA

# Install pisa
RUN pip install -e $PISA

# Install Jupyter notebook
RUN pip install notebook

# Expose the Jupyter server port
EXPOSE 8888
