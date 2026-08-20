#!/usr/bin/env sh
set -eu

SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_BIN="${SELF_DIR}/plascan"
LIB_DIR="${SELF_DIR}/../lib"

if [ ! -x "${APP_BIN}" ]; then
  APP_BIN="/opt/plascan/bin/plascan"
  LIB_DIR="/opt/plascan/lib"
fi

unset LD_LIBRARY_PATH QT_PLUGIN_PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV

export LD_LIBRARY_PATH="${LIB_DIR}:/usr/lib/x86_64-linux-gnu"
export QT_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt6/plugins"

exec "${APP_BIN}" "$@"
