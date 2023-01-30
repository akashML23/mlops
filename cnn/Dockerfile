FROM python:3.9-slim

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt
COPY ./models /app/models
# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY ./flask_app.py /app/flask_app.py

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["flask_app.py" ]