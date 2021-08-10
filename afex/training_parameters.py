from collections import namedtuple


TrainingParameters = namedtuple('TrainingParameters', [
    'epochs',
    'batch_size',
    'lr',
    'use_weights',
    'fnn_l2',
    'feature_weights_l1',
    'verbose',
    'use_surrogate_model'
])

TrainingParameters.__new__.__defaults__ = (False,) * len(TrainingParameters._fields)
