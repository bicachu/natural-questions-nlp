# Docker file for submission of the BERT-joint baseline to the Natural Questions
# competition site: https://ai.google.com/research/NaturalQuestions/competition.

# use tensorflow version 1.15 and python 3
FROM tensorflow/tensorflow:1.15.0-gpu-py3

# Upgrade pip to avoid errors
RUN pip install --upgrade pip

# Install tqdm
RUN pip install --trusted-host pypi.python.org tqdm
# Install the BERT and Natural Questions libraries.
RUN pip install --trusted-host pypi.python.org bert-tensorflow
# install with --no-dependencies to avoid wsgiref error
RUN pip install --trusted-host pypi.python.org natural-questions --no-dependencies

# Add everything in the current directory to a /nq_ensemble_model directory in the
# Docker container.
ADD . /nq_model