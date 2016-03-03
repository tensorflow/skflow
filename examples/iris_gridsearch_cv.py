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

from sklearn import datasets, metrics, grid_search, cross_validation

import skflow

# Load dataset.
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

# Fit and cross-validate DNN classifiers with 1-4 layers.
# Use fit_params to specify a directory for logging output (fit_params could
# also be used to specify a training monitor).
param_grid = [{'hidden_units':
               [[10], [10, 10], [10, 10, 10], [10, 10, 10, 10]]}]
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10],
    n_classes=3, steps=200)
fit_params = {"logdir": "example_logs"}
gs = grid_search.GridSearchCV(classifier, param_grid,
                              fit_params=fit_params,
                              scoring='accuracy')
gs.fit(X_train, y_train)

# Fit and print the best score and corresponding parameters.
print('best CV accuracy from grid search: {0:f}'.format(gs.best_score_))
print('corresponding parameters: {}'.format(gs.best_params_))

score_test = metrics.accuracy_score(y_test, gs.predict(X_test))
print('accuracy on held-out data: {0:f}'.format(score_test))
