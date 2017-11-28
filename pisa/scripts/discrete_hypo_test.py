"""
Performe discrete hyop test (standard analysis) based on command line args.
Ment to be called from `pisa.scripts.analysis` as a subcommand.
"""


from __future__ import absolute_import, division

from pisa.analysis.hypo_testing import HypoTesting
from pisa.utils.scripting import normcheckpath


__all__ = ['discrete_hypo_test']

__author__ = 'S. Wren'


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
                     ' data (aka psuedodata)')
    )

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['h0', 'h1', 'data']:
        filenames = init_args_d.pop(maker + '_pipeline')
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        init_args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        ps_str = init_args_d[ps_name]
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    # Run the analysis
    hypo_testing.run_analysis()

    if return_outputs:
        return hypo_testing
