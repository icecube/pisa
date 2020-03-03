"""PISA data container"""

from __future__ import absolute_import, division, print_function

import argparse
from collections.abc import Mapping, Iterable
from collections import OrderedDict
import copy

import numpy as np

from pisa import FTYPE
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.fileio import from_file
from pisa.utils.log import logging


__all__ = [
    "NU_FLAVORS",
    "NU_INTERACTIONS",
    "OUTPUT_NUFLAVINT_KEYS",
    "LEGACY_FLAVKEY_XLATION",
    "EventsPi",
    "split_nu_events_by_flavor_and_interaction",
    "fix_oppo_flux",
    "main",
]

__author__ = "T. Stuttard"

__license__ = """Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


# Define the flavors and interactions for neutrino events
NU_FLAVORS = OrderedDict(
    nue=12, nuebar=-12, numu=14, numubar=-14, nutau=16, nutaubar=-16
)
NU_INTERACTIONS = OrderedDict(cc=1, nc=2)
OUTPUT_NUFLAVINT_KEYS = tuple(
    "%s_%s" % (fk, ik)
    for fk, fc in NU_FLAVORS.items()
    for ik, ic in NU_INTERACTIONS.items()
)
LEGACY_FLAVKEY_XLATION = dict(
    nue="nue",
    nuebar="nuebar",
    nue_bar="nuebar",
    numu="numu",
    numubar="numubar",
    numu_bar="numubar",
    nutau="nutau",
    nutaubar="nutaubar",
    nutau_bar="nutaubar",
)


class EventsPi(OrderedDict):
    """
    Container for events for use with PISA pi

    Parameters
    ----------
    name : string, optional
        Name to identify events

    neutrinos : bool, optional
        Flag indicating if events represent neutrinos; toggles special
        behavior such as splitting into nu/nubar and CC/NC. Default is True.

    fraction_events_to_keep : float
        Fraction of loaded events to use (use to downsample).
        Must be in range [0.,1.], or disable by setting to `None`.
        Default in None.

    *args, **kwargs
        Passed on to `__init__` method of OrderedDict

    """

    def __init__(
        self,
        *args,
        name=None,
        neutrinos=True,
        fraction_events_to_keep=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.name = name
        self.neutrinos = neutrinos
        self.fraction_events_to_keep = fraction_events_to_keep

        # Check `fraction_events_to_keep` value is required range
        if self.fraction_events_to_keep is not None:
            assert (self.fraction_events_to_keep >= 0.) and (self.fraction_events_to_keep <= 1.), "`fraction_events_to_keep` must be in range [0.,1.], or None to disable"

        # Define some metadata
        self.metadata = OrderedDict(
            [
                ("detector", ""),
                ("geom", ""),
                ("runs", []),
                ("proc_ver", ""),
                ("cuts", []),
            ]
        )

    def load_events_file(self, events_file, variable_mapping=None):
        """Fill this events container from an input HDF5 file filled with event
        data Optionally can provide a variable mapping so select a subset of
        variables, rename them, etc.

        Parameters
        ----------
        events_file : string or mapping
            If string, interpret as a path and load file at that path; the
            loaded object should be a mapping. If already a mapping, take and
            interpret events from that.

        variable_mapping : mapping, optional
            If specified, should be a mapping where the keys are the
            destination variable names and the items are either the source
            variable names or an iterable of source variables names. In the
            latter case, each of the specified source variables will become a
            column vector in the destination array.

        """
        # Validate `events_file`
        if not isinstance(events_file, (str, Mapping)):
            raise TypeError(
                "`events_file` must be either string or mapping; got (%s)"
                % type(events_file)
            )

        # Validate `variable_mapping`
        if variable_mapping is not None:
            if not isinstance(variable_mapping, Mapping):
                raise TypeError("'variable_mapping' must be a mapping (e.g., dict)")
            for dst, src in variable_mapping.items():
                if not isinstance(dst, str):
                    raise TypeError("`variable_mapping` 'dst' (key) must be a string")

                if isinstance(src, str):
                    pass  # Nothing to do
                elif isinstance(src, Iterable):
                    for v in src:
                        if not isinstance(v, str):
                            raise TypeError(
                                "`variable_mapping` 'src' (value) has at least"
                                " one element that is not a string"
                            )
                else:
                    raise TypeError(
                        "`variable_mapping` 'src' (value) must be a string or"
                        " an iterable of strings"
                    )

        if isinstance(events_file, str):
            input_data = from_file(events_file)
            if not isinstance(input_data, Mapping):
                raise TypeError(
                    'Contents loaded from "%s" must be a mapping; got: %s'
                    % (events_file, type(input_data))
                )
        else:  # isinstance(events_file, Mapping)
            input_data = events_file

        # Events and EventsPi objects have attr `metadata`
        metadata = getattr(input_data, 'metadata', None)

        # HDF files have attr `attrs` attached, if present (see pisa.utils.hdf)
        if not metadata:
            metadata = getattr(input_data, 'attrs', None)

        if metadata:
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    "metadata or attrs expected to be a Mapping, but got {}".format(
                        type(metadata)
                    )
                )
            # TODO: events.py calls `tolist` method on all values that have
            # that method (e.g., convert numpy arrays to lists). Why? Is this
            # necessary? Should we do that here, too?
            self.metadata.update(metadata)

        #
        # Re-format inputs
        #

        # The following is intended to re-format input data into the desired
        # format. This is required to handle various inout cases and to ensure
        # backwards compatibility with older input file formats.

        # Convert to the required event keys, e.g. "numu_cc", "nutaubar_nc", etc.
        if self.neutrinos:
            input_data = split_nu_events_by_flavor_and_interaction(input_data)

        # The value for each category should itself be a dict of the event
        # variables, where each entry is has a variable name as the key and an
        # np.array filled once per event as the value.
        #
        # For backwards compatibility, convert to this format from known older
        # formats first
        if self.neutrinos:
            for key, cat_dict in input_data.items():
                if not isinstance(cat_dict, Mapping):
                    raise Exception(
                        "'%s' input data is not a mapping, unknown format (%s)"
                        % (key, type(cat_dict))
                    )
                for var_key, var_data in cat_dict.items():
                    if not isinstance(var_data, np.ndarray):
                        raise Exception(
                            "'%s/%s' input data is not a numpy array, unknown"
                            " format (%s)" % (key, var_key, type(var_data))
                        )

        # Ensure backwards compatibility with the old style "oppo" flux
        # variables
        if self.neutrinos:
            fix_oppo_flux(input_data)

        #
        # Load the event data
        #

        # Should be organised under a single layer of keys, each representing
        # some category of input data

        # Loop over the input types
        for data_key in input_data.keys():
            if data_key in self:
                raise ValueError(
                    "Key '%s' has already been added to this data structure"
                )

            self[data_key] = OrderedDict()

            # Loop through variable mapping
            # If none provided, just use all variables and keep the input names
            if variable_mapping is None:
                variable_mapping_to_use = tuple(
                    zip(input_data[data_key].keys(), input_data[data_key].keys())
                )
            else:
                variable_mapping_to_use = variable_mapping.items()

            # Get the array data (stacking if multiple input variables defined)
            # and check the variable exists in the input data
            for var_dst, var_src in variable_mapping_to_use:
                # TODO What about non-float data? Use dtype...
                array_data = None
                if isinstance(var_src, str):
                    var_src = [var_src]

                array_data_to_stack = []
                for var in var_src:
                    if var in input_data[data_key]:
                        array_data_to_stack.append(
                            input_data[data_key][var].astype(FTYPE)
                        )
                    else:
                        raise KeyError(
                            "Variable '%s' cannot be found for '%s' events"
                            % (var, data_key)
                        )

                # Note `squeeze` removes the extraneous 2nd dim in case of a
                # single `src`
                array_data = np.squeeze(np.stack(array_data_to_stack, axis=1))

                # Add each array to the event
                # TODO Memory copies?
                if array_data is None:
                    raise ValueError(
                        "Cannot find source variable(s) '%s' for '%s'"
                        % (var_src, data_key)
                    )
                else:
                    # Down sample events if required
                    if self.fraction_events_to_keep is not None:
                        rand = np.random.RandomState(123456) # Enforce same sample each time
                        num_events_to_keep = int(np.round(self.fraction_events_to_keep*float(array_data.size)))
                        array_data = rand.choice(array_data, size=num_events_to_keep, replace=False)

                    # Add to array
                    self[data_key][var_dst] = array_data

    def apply_cut(self, keep_criteria):
        """Apply a cut by specifying criteria for keeping events. The cut must
        be successfully applied to all flav/ints in the events object before
        the changes are kept, otherwise the cuts are reverted.

        Parameters
        ----------
        keep_criteria : string
            Any string interpretable as numpy boolean expression.

        Examples
        --------
        Keep events with true energies in [1, 80] GeV (note that units are not
        recognized, so have to be handled outside this method)

        >>> events = events.apply_cut("(true_energy >= 1) & (true_energy <= 80)")

        Do the opposite with "~" inverting the criteria

        >>> events = events.apply_cut("~((true_energy >= 1) & (true_energy <= 80))")

        Numpy namespace is available for use via `np` prefix

        >>> events = events.apply_cut("np.log10(true_energy) >= 0")

        """
        assert isinstance(keep_criteria, str)

        # Check if have already applied these cuts
        if keep_criteria in self.metadata["cuts"]:
            logging.debug(
                "Criteria '%s' have already been applied. Returning"
                " events unmodified.",
                keep_criteria,
            )
            return self

        # TODO Get everything from the GPU first ?

        # Prepare the post-cut data container
        cut_data = EventsPi(name=self.name)
        cut_data.metadata = copy.deepcopy(self.metadata)

        # Loop over the data containers
        for key in self.keys():
            cut_data[key] = {}

            # TODO Need to think about how to handle array, scalar and binned data
            # TODO Check for `events` data mode, or should this kind of logic
            # already be in the Container class?
            variables = self[key].keys()

            # Create the cut expression, and get the resulting mask
            crit_str = keep_criteria
            for variable_name in variables:
                crit_str = crit_str.replace(
                    variable_name, 'self["%s"]["%s"]' % (key, variable_name)
                )
            mask = eval(crit_str)  # pylint: disable=eval-used

            # Fill a new container with the post-cut data
            for variable_name in variables:
                cut_data[key][variable_name] = copy.deepcopy(
                    self[key][variable_name][mask]
                )

        # TODO update to GPUs?

        # Record the cuts
        cut_data.metadata["cuts"].append(keep_criteria)

        return cut_data

    def keep_inbounds(self, binning):
        """Cut out any events that fall outside `binning`. Note that events
        that fall exactly on an outer edge are kept.

        Parameters
        ----------
        binning : OneDimBinning or MultiDimBinning

        Returns
        -------
        cut_data : EventsPi

        """
        # Get the binning instance
        try:
            binning = OneDimBinning(binning)
        except:  # pylint: disable=bare-except
            pass
        if isinstance(binning, OneDimBinning):
            binning = [binning]
        binning = MultiDimBinning(binning)

        # Define a cut to remove events outside of the binned region
        bin_edge_cuts = [dim.inbounds_criteria for dim in binning]
        bin_edge_cuts = " & ".join([str(x) for x in bin_edge_cuts])

        # Apply the cut
        return self.apply_cut(bin_edge_cuts)

    def __str__(self):  # TODO Handle non-array data cases
        string = "-----------------------------\n"
        string += "EventsPi container %s :" % self.name
        for key, container in self.items():
            string += "  %s :\n" % key
            for var, array in container.items():
                array_data = array
                if len(array_data) <= 4:
                    array_data_string = str(array_data)
                else:
                    array_data_string = "[%s, %s, ..., %s, %s]" % (
                        array_data[0],
                        array_data[1],
                        array_data[-2],
                        array_data[-1],
                    )
                string += "    %s : %i elements : %s\n" % (
                    var,
                    len(array_data),
                    array_data_string,
                )
        string += "-----------------------------"
        return string


def split_nu_events_by_flavor_and_interaction(input_data):
    """Split neutrino events by nu vs nubar, and CC vs NC.

    Should be compatible with DRAGON and GRECO samples, but this depends on the
    contents of the original I3 files and whatever conversion script was used
    to produce the HDF5 files from these.

    Parameters
    ----------
    input_data : mapping

    Returns
    -------
    output_data : OrderedDict

    """
    # TODO Split into one function for nu/nubar and one for CC/NC?
    assert isinstance(input_data, Mapping)
    assert input_data, "`input_data` has no members"

    output_data = OrderedDict()

    # Loop through subcategories in the input data
    for key, data in input_data.items():
        # If key already is one of the desired keys, nothing new to do
        # Just move the data to the output container
        if key in OUTPUT_NUFLAVINT_KEYS:
            if key in output_data:
                output_data[key] = np.concatenate(output_data[key], data)
            else:
                output_data[key] = data
            continue

        # Legacy PISA HDF5 files are structured as
        #   {"<flavor>": {"<int_type>": data}};
        # and `flavor` can have "_" separating "bar". Remove such underscores
        # and flatten the nested dicts into
        #   {"<flavor>_<int_type>": data}
        # format

        if key in LEGACY_FLAVKEY_XLATION:
            new_flav_key = LEGACY_FLAVKEY_XLATION[key]
            for sub_key, sub_data in data.items():
                assert sub_key in ("cc", "nc"), str(sub_key)
                output_key = new_flav_key + "_" + sub_key
                if output_key in output_data:
                    output_data[output_key] = np.concatenate(
                        output_data[output_key], sub_data
                    )
                else:
                    output_data[output_key] = sub_data
            continue

        assert "pdg_code" in data, "No 'pdg_code' variable found for %s data" % key
        # Check these are neutrino events
        assert np.all(np.isin(data["pdg_code"], NU_FLAVORS.values())), (
            "%s data does not appear to be a neutrino data" % key
        )
        assert "interaction" in data, (
            "No 'interaction' variable found for %s data" % key
        )

        # Define a mask to select the events for each desired output key
        key_mask_pairs = [
            ("%s_%s" % (fk, ik), (data["pdg_code"] == fc) & (data["interaction"] == ic))
            for fk, fc in NU_FLAVORS.items()
            for ik, ic in NU_INTERACTIONS.items()
        ]

        # Loop over the keys/masks and write the data for each class to the
        # output container
        for mkey, mask in key_mask_pairs:
            if np.any(mask):  # Only if mask has some data
                if mkey in output_data:
                    output_data[mkey] = np.concatenate(output_data[mkey], data)
                else:
                    output_data[mkey] = data

    if len(output_data) == 0:
        raise ValueError("Failed splitting neutrino events by flavor/interaction")

    return output_data


def fix_oppo_flux(input_data):
    """Fix this `oppo` flux insanity
    someone added this in the nominal flux calculation that
    oppo flux is nue flux if flavour is nuebar, and vice versa
    here we revert that, incase these oppo keys are there

    """
    for key, val in input_data.items():
        if "neutrino_oppo_nue_flux" not in val:
            continue
        logging.warning(
            'renaming the outdated "oppo" flux keys in "%s", in the future do'
            " not use those anymore",
            key,
        )
        if "bar" in key:
            val["nominal_nue_flux"] = val.pop("neutrino_oppo_nue_flux")
            val["nominal_numu_flux"] = val.pop("neutrino_oppo_numu_flux")
            val["nominal_nuebar_flux"] = val.pop("neutrino_nue_flux")
            val["nominal_numubar_flux"] = val.pop("neutrino_numu_flux")
        else:
            val["nominal_nue_flux"] = val.pop("neutrino_nue_flux")
            val["nominal_numu_flux"] = val.pop("neutrino_numu_flux")
            val["nominal_nuebar_flux"] = val.pop("neutrino_oppo_nue_flux")
            val["nominal_numubar_flux"] = val.pop("neutrino_oppo_numu_flux")


def main():
    """Load an events file and print the contents"""
    parser = argparse.ArgumentParser(description="Events parsing")
    parser.add_argument(
        "--neutrinos",
        action="store_true",
        help="Treat input file as if it contains neutrino MC",
    )
    parser.add_argument(
        "-i", "--input-file", type=str, required=True, help="Input HDF5 events file"
    )
    args = parser.parse_args()

    events = EventsPi(neutrinos=args.neutrinos)
    events.load_events_file(args.input_file)

    logging.info("Loaded events from : %s", args.input_file)

    print("Metadata:")
    print(events.metadata)
    print(events)


if __name__ == "__main__":
    main()
