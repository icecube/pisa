# pylint: disable=not-callable
"""
Stage to evaluate the Honda flux tables using IP splines

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.flux_weights import load_2d_table, calculate_2d_flux_weights


class honda_ip(Stage):
    """
    stage to generate nominal flux

    Parameters
    ----------
    params
        Expected params .. ::
            flux_table : str

    """

    def __init__(
        self,
        **std_kwargs
    ):

        expected_params = ('flux_table',)

        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):

        self.flux_table = load_2d_table(self.params.flux_table.value)

        self.data.representation = self.calc_mode
        if self.data.is_map:
            # speed up calculation by adding links
            # as nominal flux doesn't depend on the (outgoing) flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])
        for container in self.data:
            container['nu_flux_nominal'] = np.empty((container.size, 2), dtype=FTYPE)
            container['nubar_flux_nominal'] = np.empty((container.size, 2), dtype=FTYPE)
            # container['nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)

        # don't forget to un-link everything again
        self.data.unlink_containers()

    @profile
    def compute_function(self):

        self.data.representation = self.calc_mode

        if self.data.is_map:
            # speed up calculation by adding links
            # as nominal flux doesn't depend on the (outgoing) flavour
            self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])

        # create lists for iteration
        out_names = ['nu_flux_nominal']*2 + ['nubar_flux_nominal']*2
        indices = [0, 1, 0, 1]
        tables = ['nue', 'numu', 'nuebar', 'numubar']
        for container in self.data:
            for out_name, index, table in zip(out_names, indices, tables):
                logging.info('Calculating nominal %s flux for %s', table, container.name)
                calculate_2d_flux_weights(true_energies=container['true_energy'],
                                           true_coszens=container['true_coszen'],
                                           en_splines=self.flux_table[table],
                                           out=container[out_name][:, index]
                                          )
            container.mark_changed('nu_flux_nominal')
            container.mark_changed('nubar_flux_nominal')

        # don't forget to un-link everything again
        self.data.unlink_containers()


    # def apply_function(self):

    #     self.data.representation = self.apply_mode

    #     # Set flux to be the nominal flux (choosing correct nu vs nubar flux for the container)
    #     # Note that a subsequent systematic flux stage may change this
    #     for container in self.data:
    #         np.copyto( src=container["nu%s_flux_nominal"%("" if container["nubar"] > 0 else "bar")].get("host"), dst=container["nu_flux"].get("host") )
    #         container.mark_changed('nu_flux')
