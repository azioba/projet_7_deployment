# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

#RUN apt-get update && apt-get install -y libgomp1

# prevents running sudo commands
# RUN useradd -r -s /bin/bash cazio


# Install pip requirements
COPY backend/requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY backend/app.py backend/classifier.pkl ./

# Expose port
EXPOSE 5000

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app
#CMD ["flask", "run", "--host", "0.0.0.0"]
CMD exec gunicorn --bind 0.0.0.0:5000 --workers=4 --timeout 0 app:app