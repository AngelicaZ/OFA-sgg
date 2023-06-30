# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from fairseq.X import Y`
from fairseq.fairseq.distributed import utils as distributed_utils
from fairseq.fairseq.logging import meters, metrics, progress_bar  # noqa

sys.modules["fairseq.distributed_utils"] = distributed_utils
sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar

# initialize hydra
from fairseq.fairseq.dataclass.initialize import hydra_init
hydra_init()

import fairseq.fairseq.criterions  # noqa
import fairseq.fairseq.distributed  # noqa
import fairseq.fairseq.models  # noqa
import fairseq.fairseq.modules  # noqa
import fairseq.fairseq.optim  # noqa
import fairseq.fairseq.optim.lr_scheduler  # noqa
import fairseq.fairseq.pdb  # noqa
import fairseq.fairseq.scoring  # noqa
import fairseq.fairseq.tasks  # noqa
import fairseq.fairseq.token_generation_constraints  # noqa

import fairseq.fairseq.benchmark  # noqa
import fairseq.fairseq.model_parallel  # noqa
