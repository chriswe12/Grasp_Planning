#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

docker build \
    --build-arg ISAACLAB_BASE_IMAGE=isaac-lab-base:latest \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${EULER_DOCKER_IMAGE}" \
    "${SCRIPT_DIR}"

# The upstream Isaac Sim image has /isaac-sim/runheadless.sh as its Docker
# entrypoint. Override it here so this is a Python validation, rather than an
# accidental launch of the default simulator application.
docker run --rm --entrypoint /isaac-sim/python.sh \
    "${EULER_DOCKER_IMAGE}" \
    -c "import isaaclab; print('isaaclab import OK:', isaaclab.__file__)"

docker run --rm --entrypoint /isaac-sim/python.sh \
    "${EULER_DOCKER_IMAGE}" \
    -m pip show isaaclab flatdict prettytable warp-lang
