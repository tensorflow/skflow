"""Deep Neural Network estimators."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import division, print_function, absolute_import

from sklearn.base import ClassifierMixin, RegressorMixin

from skflow.estimators.base import TensorFlowEstimator, ESTIMATOR_COMMON_DOCSTRING
from skflow import models
from skflow.util.doc_utils import Appender


class TensorFlowDNNClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow DNN Classifier model."""
    @Appender(ESTIMATOR_COMMON_DOCSTRING, join='\n')
    def __init__(self, hidden_units, n_classes, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False,
                 num_cores=4, verbose=1, early_stopping_rounds=None,
                 max_to_keep=5, keep_checkpoint_every_n_hours=10000):
        self.hidden_units = hidden_units
        super(TensorFlowDNNClassifier, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training, verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            num_cores=num_cores, max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_dnn_model(self.hidden_units,
                                    models.logistic_regression)(X, y)


class TensorFlowDNNRegressor(TensorFlowEstimator, RegressorMixin):
    """TensorFlow DNN Regressor model."""
    @Appender(ESTIMATOR_COMMON_DOCSTRING, join='\n')
    def __init__(self, hidden_units, n_classes=0, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False,
                 num_cores=4, verbose=1, early_stopping_rounds=None,
                 max_to_keep=5, keep_checkpoint_every_n_hours=10000):
        self.hidden_units = hidden_units
        super(TensorFlowDNNRegressor, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training, verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            num_cores=num_cores, max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_dnn_model(self.hidden_units,
                                    models.linear_regression)(X, y)
