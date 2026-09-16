ARG ISAACLAB_BASE_IMAGE=isaac-lab-base:latest
FROM ${ISAACLAB_BASE_IMAGE}

# The upstream image installer can continue after one extension installation
# fails. Install the core package explicitly so its dependencies and editable
# package metadata cannot be omitted from the Euler image.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-build-isolation \
        flatdict==4.0.1 && \
    /isaac-sim/python.sh -m pip install --no-cache-dir --editable \
        /workspace/isaaclab/source/isaaclab && \
    /isaac-sim/python.sh -m pip show isaaclab flatdict prettytable warp-lang
