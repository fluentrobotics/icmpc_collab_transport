#! /usr/bin/env bash

set -o errexit
set -o xtrace

pwd

cd data/fluentrobotics
pwd

tar xf rosbags.tar.zst
xxhsum -cq XXHSUMS.txt
