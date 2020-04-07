# pylint: disable=not-callable
"""
Stage to evaluate the Honda flux tables using IP splines

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.flux_weights import load_2d_table, calculate_2d_flux_weights


class pi_honda_ip(PiStage):
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
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):

        expected_params = ('flux_table',)
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_calc_keys = ()

        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('nu_flux_nominal', 'nubar_flux_nominal')

        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('nu_flux_nominal', 'nubar_flux_nominal')

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
            input_calc_keys=input_calc_keys,
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is None
        assert self.calc_mode is not None
        assert self.output_mode is not None

    def setup_function(self):

        self.flux_table = load_2d_table(self.params.flux_table.value)

        self.data.data_specs = self.calc_specs
        if self.calc_mode == 'binned':
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

        self.data.data_specs = self.calc_specs

        if self.calc_mode == 'binned':
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
                calculate_2d_flux_weights(true_energies=container['true_energy'].get('host'),
                                           true_coszens=container['true_coszen'].get('host'),
                                           en_splines=self.flux_table[table],
                                           out=container[out_name].get('host')[:, index]
                                          )
            container['nu_flux_nominal'].mark_changed('host')
            container['nubar_flux_nominal'].mark_changed('host')

        # don't forget to un-link everything again
        self.data.unlink_containers()


    # def apply_function(self):

    #     self.data.data_specs = self.output_specs

    #     # Set flux to be the nominal flux (choosing correct nu vs nubar flux for the container)
    #     # Note that a subsequent systematic flux stage may change this
    #     for container in self.data:
    #         np.copyto( src=container["nu%s_flux_nominal"%("" if container["nubar"] > 0 else "bar")].get("host"), dst=container["nu_flux"].get("host") )
    #         container['nu_flux'].mark_changed('host')
