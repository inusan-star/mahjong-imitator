#!/bin/sh
set -e

mkdir -p /app/data
chown -R ${UID}:${GID} /app/data

exec gosu user "$@"