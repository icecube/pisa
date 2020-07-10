"""
Make K-folds of data.

This stage can be used to split MC into chunks of equal size and to select only one 
chunk to make histograms from. It uses the KFold class from scikit-learn to make
"test" and "train" indeces for the dataset and sets all weights in the "train" 
indeces to zero. Optionally, weights can be re-scaled by the number of splits to
renormalize the total rates.
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer

__author__ = "A. Trettin"

__license__ = """Copyright (c) 2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


class pi_kfold(PiStage):
    """
    Stage to make splits of the MC set and select one split to make histograms.
    The weight of all indeces not belonging to the selected split are set to 
    zero.
    
    Parameters
    ----------
    n_splits (int): number of splits
    select_split (int, optional): which split to keep
    seed (int, optional): seed for the random number generator
    renormalize (bool, optional): renormalize weights by multiplying
        by the number of splits
    shuffle (bool, optional): shuffle indeces before splitting
    
    """

    def __init__(
        self,
        n_splits,
        select_split=0,
        seed=None,
        renormalize=False,
        shuffle=False,
        save_mask=False,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):

        expected_params = ()
        input_names = ()
        output_names = ()

        input_apply_keys = ("weights",)

        # The weights are simply scaled by the earth survival probability
        output_calc_keys = ("fold_weight",)
        output_apply_keys = ("weights",)

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode == "events"
        assert self.output_mode is not None

        self.n_splits = int(n_splits)
        self.select_split = int(select_split)
        if seed is None:
            self.seed = None
        else:
            self.seed = int(seed)
        self.renormalize = bool(renormalize)
        self.shuffle = bool(shuffle)

        self.save_mask = save_mask

    def setup_function(self):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
        for container in self.data:
            index_gen = kf.split(container["weights"])  # a generator
            for i, (train_index, test_index) in enumerate(index_gen):
                select_idx = test_index
                if i == self.select_split:
                    break
            container["fold_weight"] = np.zeros((container.size), dtype=FTYPE)
            select_weight = (
                kf.get_n_splits(container["weights"]) if self.renormalize else 1.0
            )
            container["fold_weight"][select_idx] = select_weight
            container["fold_weight"].mark_changed(WHERE)

            if self.save_mask:
                container['kfold_mask'] = np.zeros((container.size), dtype=np.bool)
                container['kfold_mask'][select_idx] = 1
                container['kfold_mask'].mark_changed(WHERE)


    def apply_function(self):
        for container in self.data:
            vectorizer.imul(vals=container["fold_weight"], out=container["weights"])
