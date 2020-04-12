#!/bin/sh

# Fail on the first error
set -e

# Show every execution step
set -x


case "$TASK" in
    "lint")
        pip install pylint
    ;;

    "nosetests")
        # Create virtual env using system numpy and scipy
        deactivate || true
        case "$TRAVIS_PYTHON_VERSION" in
            "2.7")
                virtualenv --system-site-packages testenv
            ;;
            "3.5")
                virtualenv -p python3.5 --system-site-packages testenv
            ;;
            "3.7")
                virtualenv -p python3.7 --system-site-packages testenv
            ;;
        esac
        source testenv/bin/activate

        # Install dependencies
        pip install --upgrade pip
        pip install numpy
        pip install scipy
        pip install pandas
        pip install scikit-learn
        pip install toolz
        pip install dask

        # Install TensorFlow
        case "$TRAVIS_OS_NAME" in
            "linux")
                case "$TRAVIS_PYTHON_VERSION" in
                    "2.7")
                        TENSORFLOW_PACKAGE_URL="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl"
                    ;;
                    "3.5")
                        TENSORFLOW_PACKAGE_URL="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc0-cp35-cp35m-linux_x86_64.whl"
                    ;;
                    "3.7")
                        TENSORFLOW_PACKAGE_URL="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.13.1-cp37-cp37m-linux_x86_64.whl"
                    ;;
                esac
            ;;
            "osx")
                TENSORFLOW_PACKAGE_URL="https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py2-none-any.whl"
            ;;
        esac
        pip install "$TENSORFLOW_PACKAGE_URL"  --ignore-installed six

        # Install test tools
        pip install codecov
        pip install coverage
        pip install nose

        # Install skflow
        python setup.py install
    ;;

esac
