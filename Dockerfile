FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /model

COPY review_polarity.tar.gz ./review_polarity.tar.gz
COPY contractions.csv ./contractions.csv
COPY train_bert.py ./train_bert.py
COPY tokenization.py ./tokenization.py

CMD python -W ignore train_bert.py