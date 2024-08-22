'''
PISA module to prep incoming data into formats that are
compatible with the mc_uncertainty likelihood formulation

This module takes in events containers from the pipeline, and
introduces an additional array giving the indices where each
event falls into.

module structure imported from bootcamp example
'''

from __future__ import absolute_import, print_function, division

__author__ = "Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)"

# Load the modified index lookup function
from pisa.core.bin_indexing import lookup_indices
from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa_tests.test_services import TEST_BINNING

__all__ = ['add_indices']


class add_indices(Stage):  # pylint: disable=invalid-name
    """
    PISA Pi stage to map out the index of the analysis
    binning where each event falls into.

    Parameters
    ----------
    params
        foo : Quantity
        bar : Quanitiy with time dimension

    Notes:
    ------
    - input and calc specs are predetermined in the module
        (inputs from the config files will be disregarded)

    - stage appends an array quantity called bin_indices
    - stage also appends an array mask to access events by
      bin index later in the pipeline

    """

    # this is the constructor with default arguments

    def __init__(self,
                 **std_kwargs,
                 ):


        # init base class
        super().__init__(
            expected_params=(),
            expected_container_keys=(),
            **std_kwargs,
        )


    def setup_function(self):
        '''
        Calculate the bin index where each event falls into

        Create one mask for each analysis bin.
        '''

        if self.calc_mode != 'events':
            raise ValueError('calc mode must be set to "events" for this module')

        if not isinstance(self.apply_mode, MultiDimBinning):
            raise ValueError('apply mode must be set to a binning')

        for container in self.data:
            self.data.representation = self.calc_mode
            variables_to_bin = []
            for bin_name in self.apply_mode.names:
                variables_to_bin.append(container[bin_name])

            indices = lookup_indices(sample=variables_to_bin,
                                     binning=self.apply_mode)

            container['bin_indices'] = indices

            self.data.representation = self.apply_mode
            for bin_i in range(self.apply_mode.tot_num_bins):
                container['bin_{}_mask'.format(bin_i)] = container['bin_indices'] == bin_i


def init_test(**param_kwargs):
    """Instantiation example"""
    return add_indices(calc_mode='events', apply_mode=TEST_BINNING)
