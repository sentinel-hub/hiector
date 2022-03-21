FROM rayproject/ray:latest-cpu

LABEL maintainer="Sinergise EO research team <eoresearch@sinergise.com>"
ARG S3_AWS_ACCESS_KEY
ARG S3_AWS_SECRET_KEY
ARG SH_INSTANCE_ID
ARG SH_CLIENT_ID
ARG SH_CLIENT_SECRET
ARG SENTINELHUB_BRANCH
ARG EOLEARN_BRANCH
ARG LCMS_BRANCH
ARG HIECTOR_BRANCH

RUN sudo apt-get update && sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
RUN sudo apt-get update && sudo apt-get install -y \
        libspatialindex-dev gcc libgeos-c1v5 libgeos-dev curl vim \
        s3fs cmake graphviz proj-bin libproj-dev gdal-bin libgdal-dev build-essential python3-opencv \
    && sudo apt-get clean && sudo apt-get autoremove -y && sudo rm -rf /var/lib/apt/lists/*

RUN pip install pip --upgrade --no-cache-dir
RUN pip install awscli wandb boto3 ipdb --upgrade --no-cache-dir
RUN pip install rasterio fiona pyproj rtree --no-cache-dir
RUN pip install gdal==$(gdalinfo --version | awk -F "," '{print $1}' | awk '{print $2}') --no-cache-dir

RUN conda clean --all --force-pkgs-dirs -y

RUN mkdir packages
WORKDIR /home/ray/packages

RUN git clone --depth 1 -b ${SENTINELHUB_BRANCH} https://github.com/sentinel-hub/sentinelhub-py.git
RUN pip install -e ./sentinelhub-py --no-cache-dir

RUN git clone --depth 1 -b ${EOLEARN_BRANCH} https://github.com/sentinel-hub/eo-learn.git
RUN pip install \
     -e ./eo-learn/core \
     -e ./eo-learn/coregistration \
     -e ./eo-learn/features \
     -e ./eo-learn/geometry \
     -e ./eo-learn/io \
     -e ./eo-learn/mask \
     -e ./eo-learn/ml_tools \
     -e ./eo-learn/visualization --no-cache-dir

RUN pip install ray[default]

RUN git clone --depth 1 -b ${HIECTOR_BRANCH} https://github.com/sentinel-hub/hiector.git
RUN pip install -e ./hiector --no-cache-dir

WORKDIR /home/ray/packages/query-planet-ccn3/hiector/ssrdd/utils/box/ext/rbbox_overlap_cpu
RUN python setup.py build_ext --inplace

WORKDIR /home/ray/
RUN sentinelhub.config --sh_client_id ${SH_CLIENT_ID} --sh_client_secret ${SH_CLIENT_SECRET} --instance_id ${SH_INSTANCE_ID}
RUN aws --profile hiector configure set aws_access_key_id ${S3_AWS_ACCESS_KEY}
RUN aws --profile hiector configure set aws_secret_access_key ${S3_AWS_SECRET_KEY}
RUN aws --profile hiector configure set region eu-central-1

RUN mkdir data
RUN cat .aws/credentials | grep -m 2 access  | awk '{print $3}' | xargs | sed 's/ /:/g' > ~/.passwd-s3fs
RUN chmod 600 ~/.passwd-s3fs
