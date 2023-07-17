# Object Detector Dockerized

Dockerized REST API service for object detection using [YOLOS](https://huggingface.co/hustvl/yolos-tiny).

## Running the docker service

The service is offered in a docker image that can be built with:

`$ docker build -t yolos_object_detection .`

or directly pulled from [Docker Hub](https://hub.docker.com/repository/docker/xavialex/yolos_object_detection/general):

`$ docker pull xavialex/yolos_object_detection`

Once the image is built, run a container with:

`$ docker run -p 80:80 --rm --gpus all --name object_detection_container yolos_object_detection`

**Note:** This docker image assumes deployment in servers with NVIDIA hardware available.

## Running locally

To run or develop the application locally, it's recommended to install the dependencies in a new virtual environment:

1. To create a Python Virtual Environment (venv) to run the code, type:

    ```python3.11 -m venv my-env```

2. Activate the new environment:
    * Windows: ```my-env\Scripts\activate.bat```
    * macOS and Linux: ```source my-env/bin/activate``` 

3. Install all the dependencies from *requirements.txt*:

    ```pip install -r requirements.txt```

After that, run the service with:

`$ uvicorn app.main:app --reload`

## Inference process

With the service running, it can be tested within the automatically generated documentation in http://localhost/docs. The request requires a confidence threshold, the IDs of the classes of interest (person and car by default, 1 and 3 respectively, more information [here](https://huggingface.co/hustvl/yolos-tiny/blob/main/config.json)) and an image/s to process. The model used is [YOLOS](https://huggingface.co/hustvl/yolos-tiny), available through HuggingFace.