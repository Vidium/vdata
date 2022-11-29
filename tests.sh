#!/bin/bash

flake8 --config .tox.ini ./vdata
mypy --show-error-codes --config-file .mypy.ini ./vdata
pytest --cov=vdata ./tests
