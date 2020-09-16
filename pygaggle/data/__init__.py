# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from .kaggle import *
from .relevance import *
from .msmarco import *
from .trec_covid import *
