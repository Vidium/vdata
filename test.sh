#!/bin/bash

flake8 --config .tox.ini ./vdata
mypy --config-file .mypy.ini ./vdata
pylint ./vdata
pytest --cov=vdata ./tests