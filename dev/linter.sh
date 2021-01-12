#!/bin/bash -e
# Adapted from Facebook's detectron2

# Run this script at project root by "./dev/linter.sh" before you commit

vergte() {
  [ "$2" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}


formatFiles=0
if [ ! -z $1 ]; then
  if [[ $1 == "format" ]]; then
    formatFiles=1
  else
    echo "Unknown option $1"
    exit 1
  fi
fi

{
	black --version | grep "19.3b0" > /dev/null
} || {
	echo "Linter requires black==19.3b0 !"
	exit 1
}

ISORT_TARGET_VERSION="4.3.21"
ISORT_VERSION=$(isort -v | grep VERSION | awk '{print $2}')
vergte "$ISORT_VERSION" "$ISORT_TARGET_VERSION" || {
  echo "Linter requires isort>=${ISORT_TARGET_VERSION} !"
  exit 1
}

set -v

echo "Running isort ..."
if [ "$formatFiles" -eq "0" ]; then
  isort -c -sp . --atomic
else
  isort -y -sp . --atomic
fi
errCode=$?
if [ "$errCode" != 0 ]; then
  exit $errCode
fi

echo "Running black ..."
if [ "$formatFiles" -eq "0" ]; then
  black --config pyproject.toml . --check
else
  black --config pyproject.toml .
fi
errCode=$?
if [ "$errCode" != 0 ]; then
  exit $errCode
fi

echo "Running flake8 ..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi
errCode=$?
if [ "$errCode" != 0 ]; then
  exit $errCode
fi

# command -v arc > /dev/null && arc lint
