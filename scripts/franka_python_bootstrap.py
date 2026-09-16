#!/usr/bin/env python3
"""Apply CUDA preloads after isaaclab.sh replaces LD_PRELOAD with libcarb."""
import os
import sys

environment = os.environ.copy()
libraries = environment.pop('FRANKA_CUDA_PRELOAD')
existing = environment.get('LD_PRELOAD')
environment['LD_PRELOAD'] = libraries + (':' + existing if existing else '')
# Re-exec before importing torch/Isaac, retaining the complete Isaac Python setup.
os.execvpe(sys.executable, [sys.executable, *sys.argv[1:]], environment)
