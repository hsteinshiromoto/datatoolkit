# ---
# Build arguments
# ---
ARG DOCKER_PARENT_IMAGE
FROM $DOCKER_PARENT_IMAGE

# NB: Arguments should come after FROM otherwise they're deleted
ARG BUILD_DATE

# Silence debconf
ARG DEBIAN_FRONTEND=noninteractive

# Add vscode user to the container
ARG PROJECT_NAME
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# ---
# Enviroment variables
# ---
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
ENV TZ Australia/Sydney
ENV SHELL=/bin/bash
ENV PROJECT_NAME=$PROJECT_NAME
ENV HOME=/home/$PROJECT_NAME

# ---
# Setup vscode as nonroot user
# ---
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ---
# Copy Container Setup Scripts
# ---
COPY bin/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY poetry.lock /usr/local/poetry.lock
COPY pyproject.toml /usr/local/pyproject.toml

RUN chmod +x /usr/local/bin/entrypoint.sh

# Create the "home" folder
RUN mkdir -p $HOME
WORKDIR $HOME

# N.B.: Keep the order 1. entrypoint, 2. cmd
# USER $USERNAME

# Get poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="${PATH}:$HOME/.poetry/bin"
ENV PATH="${PATH}:$HOME/.local/bin"

RUN poetry config virtualenvs.create false \
    && cd /usr/local \
    && poetry install --no-interaction --no-ansi

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]