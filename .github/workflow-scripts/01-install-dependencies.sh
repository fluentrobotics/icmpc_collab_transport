#! /usr/bin/env bash

set -o errexit
set -o xtrace

pwd

uv sync --all-groups
