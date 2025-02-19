#! /usr/bin/env bash

set -e

# The CSV file exported by Qualtrics contains the full question prompt in each
# of the Likert items, which makes the file hard to read and may break CSV
# parsing. This script removes large portions of the question prompts that are
# not necessary for data analysis.

# https://stackoverflow.com/questions/9670426/perl-command-line-multi-line-replace
# TODO(elvout): Also remove double quotes to make all labels consistent?
perl -0777 -i -p -e 's/How closely.+?(?=Set \d)//gs' qualtrics.csv
perl -0777 -i -p -e 's/Please.+?(?=Set \d)//gs' qualtrics.csv
