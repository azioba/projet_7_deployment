# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim


# Keeps Python from generating .pyc files in the container
#ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
#ENV PYTHONUNBUFFERED=1

# prevents running sudo commands
# RUN useradd -r -s /bin/bash cazio


# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

#WORKDIR /app
#COPY . /app

# Expose port
#EXPOSE 5000

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["flask", "run", "0.0.0.0"]
