#!/bin/sh
cd "$(dirname "$0")" && clang -O3 -ffast-math -mcpu=apple-m1 -shared -framework Accelerate -Wno-deprecated-declarations -o libreuse_prof.dylib reuse_engine.c && \
clang -O3 -ffast-math -mcpu=apple-m1 -shared -framework Accelerate -Wno-deprecated-declarations -DNOPROF -o libreuse_engine.dylib reuse_engine.c
clang -O3 -ffast-math -mcpu=apple-m1 -DNOPROF -Wno-deprecated-declarations -o reuse reuse_cli.c reuse_engine.c \
  -framework Accelerate -framework AudioToolbox -framework CoreAudio -framework CoreFoundation
