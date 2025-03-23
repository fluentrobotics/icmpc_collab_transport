#! /usr/bin/env bash

set -o errexit
set -o xtrace

pwd

wget --no-verbose \
    -O data/fluentrobotics/rosbags.tar.zst \
    "https://www.dropbox.com/scl/fo/ruzihfyim9n5wx3kcluc6/AD-AqaxhxH4kW0vv0DrE-w4/rosbags.tar.zst?rlkey=u1d4pdei9jruil1bl4j0fxkmf&dl=1"

# Determine whether we're downloading the right file from Dropbox. Use a fast,
# non-cryptographic hash since we're not super concerned about adversarial
# attacks.
xxhsum -c <<< "XXH3 (data/fluentrobotics/rosbags.tar.zst) = b703a5104c2b1153"

wget --no-verbose \
    -O data/fluentrobotics/XXHSUMS.txt \
    "https://www.dropbox.com/scl/fo/ruzihfyim9n5wx3kcluc6/AHgrJXzVImt_YWvJnkL1QUo/XXHSUMS.txt?rlkey=u1d4pdei9jruil1bl4j0fxkmf&dl=1"
xxhsum -c <<< "XXH3 (data/fluentrobotics/XXHSUMS.txt) = e52ca92e304edfb2"

wget --no-verbose \
    -O data/fluentrobotics/data-association.csv \
    "https://www.dropbox.com/scl/fo/ruzihfyim9n5wx3kcluc6/AKt7l10Z8Zz3l3VlJnEi5Ec/data-association.csv?rlkey=u1d4pdei9jruil1bl4j0fxkmf&dl=1"
xxhsum -c <<< "XXH3 (data/fluentrobotics/data-association.csv) = 6c4577d709e1d3be"

wget --no-verbose \
    -O data/fluentrobotics/qualtrics.csv \
    "https://www.dropbox.com/scl/fo/ruzihfyim9n5wx3kcluc6/AA4Uq4EipmKnswCGw-L9lZQ/qualtrics.csv?rlkey=u1d4pdei9jruil1bl4j0fxkmf&dl=1"
xxhsum -c <<< "XXH3 (data/fluentrobotics/qualtrics.csv) = 4fd5fa5d5a61779f"
