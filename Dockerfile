FROM python:3.6-slim-buster

COPY . /app
WORKDIR /app

RUN apt-get update
RUN apt-get --yes install libgtk2.0-dev
RUN pip install -r requirements.txt

CMD [ "flask", "run" , "--host=0.0.0.0", "--port=80" ]