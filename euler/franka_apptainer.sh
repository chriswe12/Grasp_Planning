#!/usr/bin/env bash
# Apptainer --nv omits NGX on Euler; expose the matching host driver library.
set -euo pipefail
if [[ "${1:-}" != exec ]]; then
    echo 'usage: franka_apptainer.sh exec [Apptainer exec arguments...]' >&2
    exit 2
fi
shift
ngx_library="$(/sbin/ldconfig -p | awk '/libnvidia-ngx.so.1 / {print $NF; found=1} END {if (!found) exit 1}')"
ngx_library="$(readlink -f "${ngx_library}")"
test -f "${ngx_library}"
export APPTAINERENV_FRANKA_NGX_LIBRARY=/opt/franka-ngx/libnvidia-ngx.so.1
echo "[FRANKA NGX] Using host driver library ${ngx_library}"
exec apptainer exec --bind "${ngx_library}:${APPTAINERENV_FRANKA_NGX_LIBRARY}:ro" "$@"
