FROM python:3.7
LABEL maintainer="n_nedelkina@mail.ru"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8180
CMD python /app/app/run_server.py
