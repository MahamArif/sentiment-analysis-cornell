FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /model

COPY review_polarity.tar.gz ./review_polarity.tar.gz
COPY contractions.csv ./contractions.csv
COPY train_xlnet.py ./train_xlnet.py

CMD python -W ignore train_xlnet.py