FROM ubuntu:latest
RUN apt-get update -y && \apt-get install -y
RUN apt-get install -y apt-transport-https
RUN apt-get install -y python3
RUN apt-get install -y vim
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender1
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-openssl

RUN pip3 install --upgrade pip 
RUN pip3 install grpcio 
RUN pip3 install grpcio-tools 
RUN pip3 install tensorflow==2.5.0 
RUN pip3 install numpy 
RUN pip3 install flask 
RUN pip3 install matplotlib 
RUN pip3 install Flask-Cors 
RUN pip3 install scikit-learn==0.20.0 
RUN pip3 install keras 
RUN pip3 install pillow 


WORKDIR /cardamage
COPY . /cardamage

CMD ["python3", "app.py"]
