import time
from math import nan, inf
from typing import Dict, List, Tuple, Union, Optional

import keras
import numpy
from keras import Model
from keras.callbacks import Callback

from lib.clone_compiled_model import clone_compiled_model

MetricName = str


class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, verbose=1, batch_size=None, record_original_history=True,
                 record_predictions=False,
                 keep_best_model_by_metric: Optional[MetricName] = None,
                 larger_result_is_better: Optional[bool] = None,
                 evaluate_on_best_model_by_metric: bool = False,
                 keep_history=False):
        """
        :param validation_sets:
        a list of
        2-tuples ((validation_generator, validation_steps), validation_set_name) or
        3-tuples (validation_data, validation_targets, validation_set_name) or
        4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        :param keep_best_model_by_metric:
        some kind of early stopping: you specify a metric and the best model
        according to this metric will be available after training in the .best_model attribute
        note that some additional work may be required if you are using custom layers in your model,
        as this feature requires cloning the model.
        :param evaluate_on_best_model_by_metric:
        if keep_best_model_by_metric is True then this parameter determines if evaluation
        should be done on the "best" found model or the most recent one
        :param keep_history:
        whether or not to keep the history for the next training run
        """
        super(AdditionalValidationSets, self).__init__()
        if larger_result_is_better is None and keep_best_model_by_metric is not None:
            raise ValueError('If you want to keep the best model you need to specify if you '
                             'want to keep the model with the largest or smallest metric '
                             '(parameter larger_result_is_better).')
        self.keep_history = keep_history
        self.evaluate_on_best_model_by_metric = evaluate_on_best_model_by_metric
        self.keep_best_model_by_metric = keep_best_model_by_metric
        self.larger_result_is_better = larger_result_is_better
        self.record_predictions = record_predictions
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3, 4]:
                raise ValueError()
        self.epoch = []
        self.history: Dict[str, List[Union[float, Dict]]] = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.record_original_history = record_original_history
        self.best_model: Optional[keras.Model] = None
        self.best_metric = self.worst_possible_metric()
        self.t = time.clock()

    def worst_possible_metric(self):
        if self.larger_result_is_better is None:
            return None
        if self.larger_result_is_better:
            return -inf
        else:
            return inf

    def on_train_begin(self, logs=None):
        if not self.keep_history:
            self.epoch = []
            self.history = {}
        if self.keep_best_model_by_metric is not None:
            if all(not self.keep_best_model_by_metric.endswith(metric)
                   for metric in self.model.metrics_names):
                raise ValueError(f'Unknown metric name: {self.keep_best_model_by_metric}')
            self.best_metric = self.worst_possible_metric()
            if self.model_to_evaluate() is not None:
                self.best_model = keras.models.clone_model(self.model)
        else:
            self.best_model = self.model

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)

        if self.record_original_history:
            # record the same values as History() as well
            logs = logs or {}
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

        if len(self.validation_sets) == 0:
            return

        # evaluate on the additional validation sets
        model: Model = self.model_to_evaluate()
        stop_training_before = self.model.stop_training

        if self.keep_best_model_by_metric:
            if all(not self.keep_best_model_by_metric.endswith(metric)
                   for metric in self.model.metrics_names):
                raise ValueError(f'Unknown metric name: {self.keep_best_model_by_metric}')

        for validation_set in self.validation_sets:
            (validation_generator, validation_steps) = None, None
            validation_data = None
            if len(validation_set) == 2:
                (validation_generator, validation_steps), validation_set_name = validation_set
                validation_targets = None
                sample_weights = None
            elif len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            predictions = None
            if model is not None:
                if validation_generator is not None:
                    results = model.evaluate_generator(validation_generator,
                                                       validation_steps)
                    if self.record_predictions:
                        predictions = predict_generator_with_labels(model, validation_generator, validation_steps)
                else:
                    results = model.evaluate(x=validation_data,
                                             y=validation_targets,
                                             verbose=0,
                                             sample_weight=sample_weights,
                                             batch_size=self.batch_size)
                    if self.record_predictions:
                        predictions = {
                            'y_pred': list(zip(*model.predict(validation_data))),
                            'y_true': list(zip(*validation_targets)),
                            'names': list(range(len(validation_targets[0]))),
                        }
                        assert (len(predictions['y_true']) ==
                                len(predictions['y_pred']) ==
                                len(predictions['names']) ==
                                len(set(predictions['names']))), \
                            (len(predictions['y_true']),
                             len(predictions['y_pred']),
                             len(predictions['names']),
                             len(set(predictions['names'])))
            else:
                results = [nan for _ in self.model.metrics_names]
                if self.record_predictions:
                    predictions = {
                        'y_pred': [],
                        'y_true': [],
                        'names': [],
                    }

            if self.record_predictions:
                value_name = self.prefix() + validation_set_name + '_predictions'
                assert predictions is not None
                self.history.setdefault(value_name, []).append(predictions)

            for i, result in enumerate(results):
                value_name = self.prefix() + validation_set_name + '_' + self.model.metrics_names[i]
                self.history.setdefault(value_name, []).append(result)

        if self.keep_best_model_by_metric and model is not None:
            if self.best_model is None:
                self.best_model = clone_compiled_model(model)
            if self.keep_best_model_by_metric in self.history:
                last_metric = self.history[self.keep_best_model_by_metric][-1]
            elif self.prefix() + self.keep_best_model_by_metric in self.history:
                last_metric = self.history[self.prefix() + self.keep_best_model_by_metric][-1]
            else:
                raise ValueError(f'Unknown metric name: {self.keep_best_model_by_metric}. '
                                 f'Available are {set(self.history).union(k[len(self.prefix()):] for k in self.history)}')

            if last_metric is not None:
                if self.larger_result_is_better and last_metric > self.best_metric:
                    self.best_metric = last_metric
                    self.best_model.set_weights(model.get_weights())
                elif not self.larger_result_is_better and last_metric < self.best_metric:
                    self.best_metric = last_metric
                    self.best_model.set_weights(model.get_weights())

                self.best_metric = self.worst_possible_metric()
            else:
                self.best_model = self.model

        if self.verbose and model is not None:
            metric_strings = [f'{round(time.clock() - self.t)}s']
            metric_strings += [f'{metric}: {values[-1]:#.4g}'
                               for metric, values in self.history.items()
                               if metric not in logs and '_predictions' not in metric]
            print(' - '.join(metric_strings))
            # headers = []
            # table = [[]]
            # for metric, values in self.history.items():
            #     if metric in logs or '_predictions' in metric:
            #         continue
            #     headers.append(metric)
            #     table[0].append(values[-1])
            # print(lib.util.my_tabulate(table, headers=headers, tablefmt='plain'))

        self.t = time.clock()
        if stop_training_before:
            self.model.stop_training = stop_training_before

    def model_to_evaluate(self) -> Optional[Model]:
        """
        To be overridden by subclasses, i was once using this to implement
        Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407)
        which builds a separate model in callbacks, that is then also evaluated.
        If `None` is returned, no evaluation is done (in that epoch).
        """
        return self.model

    # noinspection PyMethodMayBeStatic
    def prefix(self):
        """
        To be overridden by subclasses, the value returned here will be prepended to the keys in the list of metrics
        """
        return ''

    def results(self):
        """
        I actually don't remember what this method was used for, looks like it returns the results of the last epoch only
        :return: list of pairs (set_name:str, last_results:float)
        """
        if self.history == {}:
            return None
        else:
            results: List[Tuple[str, float]] = [(key, self.history[key][len(self.history[key]) - 1]) for key in
                                                self.history]
            rs: Dict[str, float] = {key: value for (key, value) in results}
            return rs


def predict_generator_with_labels(model: Model, validation_generator, validation_steps: int):
    predictions = {
        'y_pred': [],
        'y_true': [],
        'names': [],
    }
    # offsets = []  # useful debugging information in case the assertions fail
    steps = []
    for step in range(validation_steps):
        try:
            batch = validation_generator[step]
        except TypeError as e:
            if "'generator' object is not subscriptable" in str(e):
                batch = next(validation_generator)
            else:
                raise
        try:
            names = validation_generator.last_batch_names
        except AttributeError:
            if isinstance(batch[0], numpy.ndarray):
                names = list(None for _ in range(batch[0].shape[0]))
            else:  # a list of multiple inputs, just look at the first
                names = list(None for _ in range(batch[0][0].shape[0]))
        # offset = validation_generator.last_offset
        xs = batch[0]
        ys = batch[1]
        # TODO: I do not like that we have to call both .evaluate AND .predict with the same data if we want to do this
        y_pred = model.predict(xs, batch_size=xs[0].shape[0])
        predictions['y_pred'].append(y_pred)
        predictions['y_true'].append(ys)
        assert all(name is None or name not in predictions['names']
                   for name in names), (steps,
                                        # offsets
                                        )
        predictions['names'] += names
        # offsets.append(offset)
        steps.append(step)
    reformatted_predictions = {
        'y_pred': [],
        'y_true': [],
        'names': predictions['names'],
    }
    for batch in predictions['y_pred']:
        if isinstance(batch, list):  # case of multiple outputs
            for batch_idx in range(batch[0].shape[0]):
                reformatted_predictions['y_pred'].append([output[batch_idx] for output in batch])
        elif isinstance(batch, numpy.ndarray):  # case of single outputs
            for batch_idx in range(batch.shape[0]):
                reformatted_predictions['y_pred'].append(batch[batch_idx])
        else:  # some other weird case
            raise NotImplementedError(type(batch))
    for batch in predictions['y_true']:
        if isinstance(batch, list):  # case of multiple outputs
            for batch_idx in range(batch[0].shape[0]):
                if isinstance(batch, list):
                    reformatted_predictions['y_true'].append([output[batch_idx] for output in batch])
        elif isinstance(batch, numpy.ndarray):  # case of single outputs
            for batch_idx in range(batch.shape[0]):
                reformatted_predictions['y_true'].append(batch[batch_idx])
        else:  # some other weird case
            raise NotImplementedError(type(batch))
    for idx in range(len(reformatted_predictions['names'])):
        if reformatted_predictions['names'][idx] is None:
            reformatted_predictions['names'][idx] = str(idx)
    predictions = reformatted_predictions
    assert (len(predictions['y_true']) ==
            len(predictions['y_pred']) ==
            len(predictions['names']) ==
            len(set(predictions['names']))), \
        (len(predictions['y_true']),
         len(predictions['y_pred']),
         len(predictions['names']),
         len(set(predictions['names'])),
         # offsets,
         steps)
    return predictions
