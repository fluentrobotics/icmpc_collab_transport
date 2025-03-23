#! /usr/bin/env bash

set -o errexit
set -o xtrace

pwd

du --si --max-depth 1 | sort --human-numeric-sort

df --si .
df --si --print-type --exclude-type=tmpfs
