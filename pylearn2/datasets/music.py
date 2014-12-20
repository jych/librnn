"""
.. todo::

    WRITEME
"""
__author__ = "Junyoung Chung"

import os
import numpy as np
from librnn.pylearn2.datasets.SpeechDataMixin import SpeechDataMixin
from functools import wraps
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_np_rng


class MusicData(SpeechDataMixin):

    def load_data(self, which_dataset, which_set):
        """
        which_dataset : choose between 'short' and 'long'

        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")

        # Check which_dataset
        if which_dataset not in ['midi', 'nottingham', 'muse', 'jsb']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['midi', 'nottingham', 'muse', 'jsb'].")

        assert 0
        _data_path = '${YOUR_DATAPATH}'
        if which_dataset == 'midi':
            _path = os.path.join(_data_path + "Piano-midi.de.pickle")
        elif which_dataset == 'nottingham':
            _path = os.path.join(_data_path + "Nottingham.pickle")
        elif which_dataset == 'muse':
            _path = os.path.join(_data_path + "MuseData.pickle")
        elif which_dataset == 'jsb':
            _path = os.path.join(_data_path + "JSB Chorales.pickle")
        data = np.load(_path)
        self.raw_data = data[which_set]


class MusicSequence(VectorSpacesDataset, MusicData):

    def __init__(self,
                 which_dataset,
                 which_set='train',
                 stop=None):
        """
        which_dataset : specify the dataset as 'short' or 'long'
        standardize option moves all data into the (0,1) interval.
        """
        self.load_data(which_dataset, which_set)

        if which_dataset == 'midi':
            max_label = 108
        elif which_dataset == 'nottingham':
            max_label = 93
        elif which_dataset == 'muse':
            max_label = 105
        elif which_dataset == 'jsb':
            max_label = 96

        if stop is not None:
            assert stop <= len(self.raw_data)
            self.raw_data = self.raw_data[:stop]

        source = ('features', 'targets')
        space = CompositeSpace([
            SequenceDataSpace(VectorSpace(dim=max_label)),
            SequenceDataSpace(VectorSpace(dim=max_label))
        ])

        X = np.asarray(
            [np.asarray([self.list_to_nparray(time_step, max_label) for time_step in np.asarray(self.raw_data[i][:-1])])
             for i in xrange(len(self.raw_data))]
        )
        y = np.asarray(
            [np.asarray([self.list_to_nparray(time_step, max_label) for time_step in np.asarray(self.raw_data[i][1:])])
             for i in xrange(len(self.raw_data))]
        )
        super(MusicSequence, self).__init__(data=(X, y),
                                             data_specs=(space, source))

    def list_to_nparray(self, x, dim):

        y = np.zeros((dim,), dtype=np.float32)
        for i in x:
            y[i - 1] = 1

        return y

    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):
        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    @wraps(VectorSpacesDataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode=None):
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        # This should be fixed to allow iteration with default data_specs
        # i.e. add a mask automatically maybe?
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)
