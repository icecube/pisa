"""
Performe discrete hyop test (standard analysis) based on command line args.
Ment to be called from `pisa.scripts.analysis` as a subcommand.
"""


from __future__ import absolute_import, division

from pisa.analysis.hypo_testing import HypoTesting, setup_makers_from_pipelines,\
                                       collect_maker_selections
from pisa.utils.scripting import normcheckpath


__all__ = ['discrete_hypo_test']

__author__ = 'S. Wren'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


def discrete_hypo_test(return_outputs=False):
    """Setup distribution makers and run the hypo_testing process.

    Parameters
    ----------
    return_outputs : bool
        Whether to return the hypo_testing object

    Returns
    -------
    hypo_testing : None or :class:`pisa.analysis.HypoTesting`
        If `return_outputs` is True, returns the object used for running the
        analysis (e.g. for calling this script/function from an interactive
        shell).

    """
    # NOTE: import here to avoid circular refs
    from pisa.scripts.analysis import parse_args

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.
    init_args_d = parse_args(
        command=discrete_hypo_test,
        description=('Test the ability to distinguish between two hypotheses'
                     ' based on "data": real data, toy data, or fluctuated toy'
                     ' data (aka pseudodata)')
    )

    setup_makers_from_pipelines(init_args_d, ref_maker_names=['h0', 'h1', 'data'])
    collect_maker_selections(init_args_d, maker_names=['h0', 'h1', 'data'])

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    # Run the analysis
    hypo_testing.run_analysis()

    if return_outputs:
        return hypo_testing
