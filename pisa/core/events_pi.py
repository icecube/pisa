#PISA data container

from __future__ import absolute_import, division, print_function

import copy, collections

import numpy as np 

from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.log import logging
from pisa.utils.fileio import from_file, to_file
#from pisa.core.container import Container
from pisa import FTYPE
from pisa.utils.numba_tools import WHERE



__all__ = ["EventsPi","convert_nu_data_to_flat_format"]


class EventsPi(collections.OrderedDict) :

    def __init__(self,name=None,*arg,**kw) :

        super(EventsPi, self).__init__(*arg, **kw)

        self.name = "events" if name is None else name

        # Define some metadata
        self.metadata = collections.OrderedDict([
            ('detector', ''),
            ('geom', ''),
            ('runs', []),
            ('proc_ver', ''),
            ('cuts', []),
        ])



    def load_events_file(self,events_file,variable_mapping=None) :

        # Check format of variable_mapping
        # Should be a dict, where the keys are the destination variable names and the items
        # are either the source variable names, or a list of source variables names that will be combined
        if variable_mapping is not None :
            if not isinstance(variable_mapping,collections.Mapping) :
                raise ValueError("'variable_mapping' must be a dict")
            for dst,src in variable_mapping.items() :
                if not isinstance(dst,basestring) :
                    raise ValueError("'variable_mapping' 'dst' (key) must be a string")
                if isinstance(src,basestring) :
                    pass #Nothing to do
                elif isinstance(src,collections.Iterable) :
                    for v in src :
                        if not isinstance(v,basestring) :
                            raise ValueError("'variable_mapping' 'dst' (value) has at least one element that is not a string")
                else :
                    raise ValueError("'variable_mapping' 'src' (value) must be a string, or a list of strings")

        # Open the input file
        input_data = from_file(events_file)

        # Input data should be a dict where each key is a category of data
        if not isinstance(input_data,collections.Mapping) :
            raise Exception("Input data is not a dict, unknown format (%s)" % type(input_data))

        # The value for each category should itself be a dict of the event variables,
        # where each entry is has a variable name as the key and a np.array filled once 
        # per event as the value.
        # For backwards compatibility, convert to thisfromat from knwon older formats first
        convert_nu_data_to_flat_format(input_data)
        for cat_key,cat_dict in input_data.items() :
            if not isinstance(cat_dict,collections.Mapping) :
                raise Exception("'%s' input data is not a dict, unknown format (%s)" % (cat_key,type(cat_dict)))
            for var_key,var_data in cat_dict.items() :
                if not isinstance(var_data,np.ndarray) :
                    raise Exception("'%s/%s' input data is not a numpy array, unknown format (%s)" % (cat_key,var_key,type(var_data)))

        # Ensure backwards compatibility wiht the old style "oppo" flux variables
        fix_oppo_flux(input_data)

        # Load events
        # Should be organised under a single layer of keys, each representing some cateogry of input data
        for data_key in input_data.keys() :

            if data_key in self :
                raise ValueError("Key '%s' has already been added to this data structure")

            # Create a container for this events category
            #container = Container(data_key)
            #container.data_specs = "events"

            self[data_key] = collections.OrderedDict()

            # Loop through variable mapping
            # If none provided, just use all variables and keep the input names
            variable_mapping_to_use = zip(input_data[data_key].keys(),input_data[data_key].keys()) if variable_mapping is None else variable_mapping.items()
            for var_dst,var_src in variable_mapping_to_use :

                #TODO What to do if variable doesn't exist? Right now just ignore it, but might want to raise exception. Will be complicated though by case of species with different variables (e.g. no axial mass in muons)

                # Get the array data (stacking if multiple input variables defined) #TODO What about non-float data? Use dtype...
                array_data = None
                if not np.isscalar(var_src) :
                    array_data_to_stack = [ input_data[data_key][var].astype(FTYPE) for var in var_src if var in input_data[data_key] ]
                    if len(array_data_to_stack) == len(var_src) :
                        array_data = np.stack(array_data_to_stack,axis=1)
                else :
                    if var_src in input_data[data_key] :
                        array_data = input_data[data_key][var_src].astype(FTYPE)

                # Add each array to the event #TODO Memory copies?
                if array_data is not None :
                    #container.add_array_data(var_dst,array_data) #TODO use the special cases that Philipp added to simple_data_loader
                    self[data_key][var_dst] = array_data
                else :
                    logging.warn("Source variable(s) not present for '%s', skipping mapping : %s -> %s"%(data_key,var_src,var_dst))

            #self[data_key] = container


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

        assert isinstance(keep_criteria, basestring)

        if not keep_criteria:
            logging.debug(
                'Empy criteria. Returning events unmodified.'
            )
            return self

        # Check if have already applied these cuts
        if keep_criteria in self.metadata['cuts']:
            logging.debug("Criteria '%s' have already been applied. Returning"
                          " events unmodified.", keep_criteria)
            return self

        #TODO Get everything from the GPU first ?

        # Prepare the post-cut data container
        cut_data = EventsPi(name=self.name)
        cut_data.metadata = copy.deepcopy(self.metadata)

        logging.debug(
            'Applying cuts "%s".' % keep_criteria
        )

        # Loop over the data containers
        for key in self.keys() :

            cut_data[key] = {}

            #TODO Need to think about how to handle array, scalar and binned data
            #TODO Should this kind of logic already be in the Container class?
            #if self[key].data_mode != "events" :
            #    raise ValueError("'apply_cuts' only implemented for 'events' data mode so far")
            #variables = self[key].array_data.keys()
            variables = self[key].keys()

            # Create the cut expression, and get the resulting mask
            crit_str = keep_criteria
            for variable_name in variables:
                crit_str = crit_str.replace(
                    #variable_name, 'self["%s"]["%s"].get(WHERE)'%(key,variable_name)
                    variable_name, 'self["%s"]["%s"]'%(key,variable_name)
                )
            mask = eval(crit_str)

            # Fill a new container with the post-cut data
            #cut_data[key] = Container(key)
            #cut_data[key].data_specs = self[key].data_specs
            for variable_name in variables :
                #cut_data[key].add_array_data(variable_name,copy.deepcopy(self[key][variable_name].get(WHERE)[mask]))
                cut_data[key][variable_name] = copy.deepcopy(self[key][variable_name][mask])

        #TODO update to GPUs?

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
        except:
            pass
        if isinstance(binning, OneDimBinning):
            binning = [binning]
        binning = MultiDimBinning(binning)

        # Check that the current cuts have not alrady been applied
        current_cuts = self.metadata['cuts']
        new_cuts = [dim.inbounds_criteria for dim in binning]
        unapplied_cuts = [c for c in new_cuts if c not in current_cuts]
        if len(unapplied_cuts) == 0:
            logging.debug("All inbounds criteria '%s' have already been"
                          " applied. Returning events unmodified.", new_cuts)
            return self
        all_cuts = copy.deepcopy(current_cuts) + unapplied_cuts

        # Create a single cut from all unapplied cuts
        keep_criteria = ' & '.join(['(%s)' % c for c in unapplied_cuts])

        # Do the cutting
        cut_data = self.apply_cut(keep_criteria=keep_criteria)

        # Replace the combined 'cuts' string with individual cut strings
        #cut_data.metadata['cuts'].append = all_cuts

        return cut_data


    def __str__(self) : #TODO Handle non-array data cases
        string = "-----------------------------\n"
        string += "EventsPi container %s :" % self.name
        for key,container in self.items() :
            string += "  %s :\n" % key
            #for var,array in container.array_data.items() :
            for var,array in self[key].items() :
                #array_data = array.get()
                array_data = array
                array_data_string = str(array_data) if len(array_data) <= 4 else "[%s,%s,...,%s,%s]"%(array_data[0],array_data[1],array_data[-2],array_data[-1])
                string += "    %s : %i elements : %s\n" % (var,len(array_data),array_data_string)
        string += "-----------------------------"
        return string
            



def convert_nu_data_to_flat_format(input_data) :
    """Format for events files is now a single layer of categories.
    For backwards compatibility, also want to handle case where there is an 
    additional layer between the categories and variables, as older files have
    the format [nu_flavor][nc/cc], whilst new ones are [nu_flavor_nc/cc]

    """

    int_keys = ["nc","cc"]
    for top_key,top_dict in input_data.items() :
        if set(top_dict.keys()) == set(int_keys) :
            for int_key in int_keys :
                new_top_key = top_key + "_" + int_key
                if "_bar" in top_key :
                    new_top_key = new_top_key.replace("_bar","bar") #numu_bar -> numubar conversion
                input_data[new_top_key] = top_dict.pop(int_key)
            input_data.pop(top_key)



def fix_oppo_flux(input_data) :
    """Fix this `oppo` flux insanity
    someone added this in the nominal flux calculation that
    oppo flux is nue flux if flavour is nuebar, and vice versa
    here we revert that, incase these oppo keys are there

    """

    for key,val in input_data.items():
        if 'neutrino_oppo_nue_flux' not in val:
            continue
        logging.warning('renaming the outdated "oppo" flux keys in "%s", in the future do not use those anymore'%key)
        if 'bar' in key:
            val['nominal_nue_flux'] = val.pop('neutrino_oppo_nue_flux')
            val['nominal_numu_flux'] = val.pop('neutrino_oppo_numu_flux')
            val['nominal_nuebar_flux'] = val.pop('neutrino_nue_flux')
            val['nominal_numubar_flux'] = val.pop('neutrino_numu_flux')
        else :
            val['nominal_nue_flux'] = val.pop('neutrino_nue_flux')
            val['nominal_numu_flux'] = val.pop('neutrino_numu_flux')
            val['nominal_nuebar_flux'] = val.pop('neutrino_oppo_nue_flux')
            val['nominal_numubar_flux'] = val.pop('neutrino_oppo_numu_flux')


'''
Main functions
'''

def main() :
    """This main function can be used to load an events file and print the contents"""

    import argparse
    parser = argparse.ArgumentParser(description="Events parsing")
    parser.add_argument("--input-file",type=str,required=True,help="Input HDF5 events file")
    args = parser.parse_args()

    events = EventsPi()
    events.load_events_file(args.input_file)

    print("Loaded events from : %s"%args.input_file)

    print(events)


if  __name__ == "__main__":
    main()

