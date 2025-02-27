#!/usr/bin/env bash
python3 -m coverage run --omit=tests/* -m pytest . && echo "Tests Passed" || exit 1
python3 -m coverage lcov -i -o coverage/lcov.info
