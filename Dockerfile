FROM python:3.10

# set a directory for the app
WORKDIR /usr/src/app

ENV PORT 80
# copy all the files to the container
COPY . /usr/src/app

# resolve error libgl
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install dependencies
RUN pip install -r requirements.txt

# port the app is runing on
EXPOSE 8888

# run the command
CMD python ./main.py