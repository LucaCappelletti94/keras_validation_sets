from math import nan
from typing import Dict, List, Tuple, Union, Optional

from keras import Model
from keras.callbacks import Callback

"""
This version is the most recent one that I am using at the moment, but it requires the generators to have some specific attributes that they usually dont have.
For example when storing the predictions, i also wanted to record the names of the samples so that i can later distinguish them.

For a version that only includes the parts that I consider relevant for the public (and slightly more comments), I added "relevant_parts_of_most_recent_version.py".
I never tested this other file though, could be stupid mistakes in there.
"""


class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None, record_original_history=True,
                 record_predictions=False):
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
        """
        super(AdditionalValidationSets, self).__init__()
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

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

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
                        predictions = {
                            'y_pred': [],
                            'y_true': [],
                            'names': [],
                        }
                        offsets = []  # useful debugging information in case the assertions fail
                        steps = []
                        for step in range(validation_steps):
                            batch = validation_generator[step]
                            names = validation_generator.last_batch_names
                            offset = validation_generator.last_offset
                            xs = batch[0]
                            ys = batch[1]
                            # TODO: I do not like that we have to call both .evaluate AND .predict with the same data if we want to do this
                            y_pred = model.predict(xs, batch_size=xs[0].shape[0])
                            predictions['y_pred'].append(y_pred)
                            predictions['y_true'].append(ys)
                            try:
                                assert all(name not in predictions['names']
                                           for name in names), (offsets, steps)
                                predictions['names'] += names
                            except AttributeError:
                                # just numbers starting from 0
                                predictions['names'] += [str(i + len(predictions['names'])) for i in range(len(ys))]
                            offsets.append(offset)
                            steps.append(step)
                        reformatted_predictions = {
                            'y_pred': [],
                            'y_true': [],
                            'names': predictions['names'],
                        }
                        for batch in predictions['y_pred']:
                            for batch_idx in range(batch[0].shape[0]):
                                reformatted_predictions['y_pred'].append([output[batch_idx] for output in batch])
                        for batch in predictions['y_true']:
                            for batch_idx in range(batch[0].shape[0]):
                                reformatted_predictions['y_true'].append([output[batch_idx] for output in batch])
                        predictions = reformatted_predictions
                        assert (len(predictions['y_true']) ==
                                len(predictions['y_pred']) ==
                                len(predictions['names']) ==
                                len(set(predictions['names']))), \
                            (len(predictions['y_true']),
                             len(predictions['y_pred']),
                             len(predictions['names']),
                             len(set(predictions['names'])),
                             offsets,
                             steps)
                else:
                    results = model.evaluate(x=validation_data,
                                             y=validation_targets,
                                             verbose=0,
                                             sample_weight=sample_weights,
                                             batch_size=self.batch_size)
                    if self.record_predictions:  # TODO check if this works
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
                if self.verbose == 1:
                    print(self.prefix() + validation_set_name + '_' + self.model.metrics_names[i], result)

    def model_to_evaluate(self) -> Optional[Model]:
        """
        To be overridden by subclasses
        If `None` is returned, no evaluation is done (in that epoch).
        """
        return self.model

    # noinspection PyMethodMayBeStatic
    def prefix(self):
        """
        To be overridden by subclasses
        """
        return ''

    def results(self):
        """
        :return: list of pairs (set_name:str, last_results:float)
        """
        if self.history == {}:
            return None
        else:
            results: List[Tuple[str, float]] = [(key, self.history[key][len(self.history[key]) - 1]) for key in
                                                self.history]
            rs: Dict[str, float] = {key: value for (key, value) in results}
            return rs
