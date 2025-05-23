# Use the official Ubuntu base image
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y python3.10 python3-pip

ENV PISA=/pisa
ENV PATH=$PISA/bin:$PATH

RUN mkdir -p $PISA

ADD ../docs $PISA/docs/
ADD ../images $PISA/images/
ADD ../pisa $PISA/pisa/
ADD ../pisa_examples $PISA/pisa_examples/
ADD ../pisa_tests $PISA/pisa_tests/
ADD ../README_files $PISA/README_files/

WORKDIR $PISA
RUN pip install -e $PISA

# Expose the Jupyter server port
EXPOSE 8888

# Add a custom script for starting the Jupyter server
ADD docker/start-jupyter-server-in-container.sh /usr/local/bin/start-jupyter-server
RUN chmod +x /usr/local/bin/start-jupyter-server
    
WORKDIR /workspace