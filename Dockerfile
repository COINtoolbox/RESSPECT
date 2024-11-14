FROM ubuntu

ENV RESSPECT_DIR=/resspect

# Note that this is where we copy resspect source at build time
# and also where docker-compose mounts the curent source directory in this container.
ENV RESSPECT_SRC=${RESSPECT_DIR}/resspect-src
ENV RESSPECT_VENV=${RESSPECT_DIR}/resspect-venv
ENV RESSPECT_VENV_BIN=${RESSPECT_VENV}/bin
ENV RESSPECT_WORK=${RESSPECT_DIR}/resspect-work

WORKDIR ${RESSPECT_DIR}

RUN echo "Entering resspect Dockerfile"

RUN apt-get update && \
   apt-get -y upgrade && \
   apt-get clean && \
   apt-get install -y python3 python3-pip python3-venv postgresql-client git && \
   rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy over resspect source from local checkout so we can install dependencies
# and so the container will have a version of RESSPECT packaged when run standalone.
#
# If this line is changed, pleas refer to the bind mount in the compose file to ensure consistency.
COPY . ${RESSPECT_SRC}

# Create a venv for resspect
RUN python3 -m venv ${RESSPECT_VENV}

# Use this venv for future python commands in this dockerfile
ENV PATH=${RESSPECT_VENV_BIN}:$PATH

# Activate the venv every time we log in.
RUN touch /root/.bashrc && echo "source ${RESSPECT_VENV_BIN}/activate" >> /root/.bashrc

# Install RESSPECT and its dependencies within the virtual env.
#
# We inject a pretend version number via `git rev-parse HEAD` so that pip can find a version number for 
# RESSPECT when this is built in github actions CI. Finding a version number depends deeply on available
# git metadata in the local checkout.
#
# When called from docker/build-push-action, buildkit manages its own separate checkout of the 
# container source rather than using the checkout supplied by the github actions runner.
#
# While the docker/build-push-action can be configured to retain the .git directory using 
# `build-args: BUILDKIT_CONTEXT_KEEP_GIT_DIR=1` -- and we rely on that below --
# docker/build-push-action cannot be configured to download tags.
#
# Sadly, pip uses setuptools_scm to generate a version number for our package. In turn
# setuptools_scm relies on git-describe. git-describe relies on downloaded tags in the 
# checkout to produce output which is processable by setuptools_scm, and therefore pip. 
# These tags are simly not present when using docker/build-push-action.
#
# This approach has been adapted from the workaround in the setuptools_scm documentation here:
# https://setuptools-scm.readthedocs.io/en/latest/usage/#with-dockerpodman
#
# It is probably not appropriate for publishing docker containers because it does not fall back
# to the actual version number of the package during a release build.
RUN bash -c "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RESSPECT=0+$(cd ${RESSPECT_SRC} && git rev-parse HEAD) \
             pip install ${RESSPECT_SRC}"

# Create a sample work dir for resspect
RUN mkdir -p ${RESSPECT_WORK}/results
RUN mkdir -p ${RESSPECT_WORK}/plots
RUN cp -r ${RESSPECT_SRC}/data ${RESSPECT_WORK}

EXPOSE 8081
ENTRYPOINT ["bash"]
