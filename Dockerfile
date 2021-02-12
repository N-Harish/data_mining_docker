FROM python:3.8.7

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN python -m nltk.downloader -d /app/nltk_data stopwords

RUN python -m nltk.downloader -d /app/nltk_data punkt

RUN python -m nltk.downloader -d /app/nltk_data wordnet

EXPOSE 8051

CMD streamlit run Data_Mining.py --server.port $PORT