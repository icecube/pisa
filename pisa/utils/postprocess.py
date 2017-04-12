"""
A bunch of functions useful for postprocessing scripts

"""

import re

from pisa.utils.log import logging


def tex_axis_label(label):
    '''
    Takes the labels used in the objects and turns them in to something nice
    for plotting. This can never truly be exhaustive, but it definitely does 
    the trick. If something looks ugly add it to this function!
    '''
    label = label.lower()
    pretty_labels = {}
    pretty_labels["atm_muon_scale"] = r"Muon Background Scale"
    pretty_labels["nue_numu_ratio"] = r"$\nu_e/\nu_{\mu}$ Ratio"
    pretty_labels["nu_nubar_ratio"] = r"$\nu/\bar{\nu}$ Ratio"
    pretty_labels["barr_uphor_ratio"] = r"Barr Up/Horizontal Ratio"
    pretty_labels["barr_nu_nubar_ratio"] = r"Barr $\nu/\bar{\nu}$ Ratio"
    pretty_labels["delta_index"] = r"Atmospheric Index Change"
    pretty_labels["theta13"] = r"$\theta_{13}$"
    pretty_labels["theta23"] = r"$\theta_{23}$"
    pretty_labels["sin2theta23"] = r"$\sin^2\theta_{23}$"
    pretty_labels["deltam31"] = r"$\Delta m^2_{31}$"
    pretty_labels["deltam3l"] = r"$\Delta m^2_{3l}$"
    pretty_labels["aeff_scale"] = r"$A_{\mathrm{eff}}$ Scale"
    pretty_labels["energy_scale"] = r"Energy Scale"
    pretty_labels["genie_ma_qe"] = r"GENIE $M_{A}^{QE}$"
    pretty_labels["genie_ma_res"] = r"GENIE $M_{A}^{Res}$"
    pretty_labels["dom_eff"] = r"DOM Efficiency"
    pretty_labels["hole_ice"] = r"Hole Ice"
    pretty_labels["hole_ice_fwd"] = r"Hole Ice Forward"
    pretty_labels["degree"] = r"$^\circ$"
    pretty_labels["radians"] = r"rads"
    pretty_labels["electron_volt ** 2"] = r"$\mathrm{eV}^2$"
    pretty_labels["electron_volt"] = r"$\mathrm{eV}^2$"
    pretty_labels["llh"] = r"Likelihood"
    pretty_labels["conv_llh"] = r"Convoluted Likelihood"
    pretty_labels["chi2"] = r"$\chi^2$"
    pretty_labels["mod_chi2"] = r"Modified $\chi^2$"
    pretty_labels["no"] = r"Normal Ordering"
    pretty_labels["io"] = r"Inverted Ordering"
    pretty_labels["nomsw"] = r"Normal Ordering, Matter Oscillations"
    pretty_labels["iomsw"] = r"Inverted Ordering, Matter Oscillations"
    pretty_labels["novacuum"] = r"Normal Ordering, Vacuum Oscillations"
    pretty_labels["iovacuum"] = r"Inverted Ordering, Vacuum Oscillations"
    pretty_labels["msw"] = r"Matter Oscillations"
    pretty_labels["vacuum"] = r"Vacuum Oscillations"
    pretty_labels["no,llr"] = r"LLR Method"
    pretty_labels["no,llr,nufitpriors"] = r"LLR Method, Nu-Fit Priors"
    pretty_labels["io,llr"] = r"LLR Method"
    pretty_labels["io,llr,nufitpriors"] = r"LLR Method, Nu-Fit Priors"
    pretty_labels["nue"] = r"$\nu_e$"
    pretty_labels["nuebar"] = r"$\bar{\nu}_e$"
    pretty_labels["numu"] = r"$\nu_{\mu}$"
    pretty_labels["numubar"] = r"$\bar{\nu}_{\mu}$"
    pretty_labels["second"] = r"s"
    pretty_labels["seconds"] = r"s"
    pretty_labels["atm_delta_index"] = r"Atmospheric Index Change"
    if label not in pretty_labels.keys():
        logging.warn("I don't know what to do with %s. Returning as is."%label)
        return label
    return pretty_labels[label]


def parse_pint_string(pint_string):
    '''
    Will return the value and units from a string with attached pint-style 
    units. i.e. the string "0.97 dimensionless" would return a value of 0.97 
    and units of dimensionless. Both will return as strings.
    '''
    val = pint_string.split(' ')[0]
    units = pint_string.split(val+' ')[-1]
    return val, units


def extract_gaussian(prior_string, units):
    '''
    Parses the string for the Gaussian priors that comes from the config 
    summary file in the logdir. This should account for dimensions though has 
    only been proven with "deg" and "ev ** 2".
    '''
    if units == 'dimensionless':
        parse_string = ('gaussian prior: stddev=(.*)'
                        ' , maximum at (.*)')
        bits = re.match(
            parse_string,
            prior_string,
            re.M|re.I
        )
        stddev = float(bits.group(1))
        maximum = float(bits.group(2))
    else:
        try:
            # This one works for deg and other single string units
            parse_string = ('gaussian prior: stddev=(.*) (.*)'
                            ', maximum at (.*) (.*)')
            bits = re.match(
                parse_string,
                prior_string,
                re.M|re.I
            )
            stddev = float(bits.group(1))
            maximum = float(bits.group(3))
        except:
            # This one works for ev ** 2 and other triple string units
            parse_string = ('gaussian prior: stddev=(.*) (.*) (.*) (.*)'
                            ', maximum at (.*) (.*) (.*) (.*)')
            bits = re.match(
                parse_string,
                prior_string,
                re.M|re.I
            )
            stddev = float(bits.group(1))
            maximum = float(bits.group(5))

    return stddev, maximum


def get_num_rows(data, omit_metric=False):
    '''
    Calculates the number of rows for multiplots based on the number of 
    systematics.
    '''
    if omit_metric:
        num_rows = int((len(data.keys())-1)/4)
    else:
        num_rows = int(len(data.keys())/4)
    if len(data.keys())%4 != 0:
        num_rows += 1
    return num_rows


def parse_binning_string(binning_string):
    '''
    Returns a dictionary that can be used to instantiate a binning object from
    the output of having run str on the original binning object.
    '''
    if 'MultiDimBinning' in binning_string:
        raise ValueError('This function is designed to work with OneDimBinning'
                         ' objects. You should separate the MultiDimBinning '
                         'string in to the separate OneDimBinning strings '
                         'before calling this function and then reconnect them'
                         ' in to the MultiDimBinning object after.')
    if 'OneDimBinning' not in binning_string:
        raise ValueError('String expected to have OneDimBinning in it. Got %s'
                         %binning_string)
    binning_dict = {}
    if '1 bin ' in binning_string:
        raise ValueError('Singular bin case not dealt with yet')
    elif 'irregularly' in binning_string:
        parse_string = ('OneDimBinning\((.*), (.*) irregularly-sized ' + \
                        'bins with edges at \[(.*)\] (.*)\)')
        a = re.match(parse_string, binning_string, re.M|re.I)
        # Match should come out None is the bins don't have units
        if a is None:
            parse_string = ('OneDimBinning\((.*), (.*) irregularly-sized ' + \
                            'bins with edges at \[(.*)\]\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
        else:
            binning_dict['units'] = a.group(4)
        binning_dict['name'] = a.group(1).strip('\'')
        binning_dict['num_bins'] = int(a.group(2))
        binning_dict['bin_edges'] = [float(i) for i in a.group(3).split(', ')]
    elif 'logarithmically' in binning_string:
        parse_string = ('OneDimBinning\((.*), (.*) logarithmically-uniform ' + \
                        'bins spanning \[(.*), (.*)\] (.*)\)')
        a = re.match(parse_string, binning_string, re.M|re.I)
        # Match should come out None is the bins don't have units
        if a is None:
            parse_string = ('OneDimBinning\((.*), (.*) logarithmically' + \
                            '-uniform bins spanning \[(.*), (.*)\]\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
        else:
            binning_dict['units'] = a.group(5)
        binning_dict['name'] = a.group(1).strip('\'')
        binning_dict['num_bins'] = int(a.group(2))
        binning_dict['domain'] = [float(a.group(3)), float(a.group(4))]
        binning_dict['is_log'] = True
    elif 'equally-sized' in binning_string:
        parse_string = ('OneDimBinning\((.*), (.*) equally-sized ' + \
                        'bins spanning \[(.*) (.*)\] (.*)\)')
        a = re.match(parse_string, binning_string, re.M|re.I)
        # Match should come out None is the bins don't have units
        if a is None:
            parse_string = ('OneDimBinning\((.*), (.*) equally-sized ' + \
                            'bins spanning \[(.*), (.*)\]\)')
            a = re.match(parse_string, binning_string, re.M|re.I)
        else:
            binning_dict['units'] = a.group(5)
        binning_dict['name'] = a.group(1).strip('\'')
        binning_dict['num_bins'] = int(a.group(2))
        binning_dict['domain'] = [float(a.group(3)), float(a.group(4))]
        binning_dict['is_lin'] = True

    add_tex_to_binning(binning_dict)
    return binning_dict


def add_tex_to_binning(binning_dict):
    '''
    Will add a tex to binning dictionaries parsed with the above function.
    '''
    if 'reco' in binning_dict['name']:
        sub_string = 'reco'
    elif 'true' in binning_dict['name']:
        sub_string = 'true'
    else:
        sub_string = None
    if 'energy' in binning_dict['name']:
        binning_dict['tex'] = r'$E_{%s}$'%sub_string
    elif 'coszen' in binning_dict['name']:
        binning_dict['tex'] = r'$\cos\theta_{Z,%s}$'%sub_string


def plot_colour(label):
    '''
    Will return a standard colour scheme which can be used for e.g. specific 
    truths or specific ice models etc.
    '''
    label = label.lower()
    pretty_colours = {}
    # SPIce HD
    pretty_colours['544'] = 'maroon'
    pretty_colours['545'] = 'goldenrod'
    pretty_colours['548'] = 'blueviolet'
    pretty_colours['549'] = 'forestgreen'
    # H2
    ## DOM Efficiency Sets
    pretty_colours['551'] = 'cornflowerblue'
    pretty_colours['552'] = 'cornflowerblue'
    pretty_colours['553'] = 'cornflowerblue'
    pretty_colours['554'] = 'mediumseagreen'
    pretty_colours['555'] = 'mediumseagreen'
    pretty_colours['556'] = 'mediumseagreen'
    ## Hole Ice Sets
    pretty_colours['560'] = 'olive'
    pretty_colours['561'] = 'olive'
    pretty_colours['564'] = 'darkorange'
    pretty_colours['565'] = 'darkorange'
    pretty_colours['572'] = 'teal'
    pretty_colours['573'] = 'teal'
    ## Dima Hole Ice Set without RDE
    pretty_colours['570'] = 'mediumvioletred'
    
    ## Baseline
    pretty_colours['585'] = 'slategrey'

    for colourkey in pretty_colours.keys():
        if colourkey in label:
            return pretty_colours[colourkey]
    else:
        logging.info("I do not have a colour scheme for your label %s. "
                     "Returning black."%label)
        return 'k'

def plot_style(label):
    '''
    Will return a standard line style for plots similar to above.
    '''
    label = label.lower()
    pretty_styles = {}
    # H2
    ## DOM Efficiency Sets
    pretty_styles['552'] = '--'
    pretty_styles['553'] = '-.'
    pretty_styles['555'] = '--'
    pretty_styles['556'] = '-.'
    ## Hole Ice Sets
    pretty_styles['561'] = '--'
    pretty_styles['565'] = '--'
    pretty_styles['572'] = '--'
    pretty_styles['573'] = '-.'

    for colourkey in pretty_styles.keys():
        if colourkey in label:
            return pretty_styles[colourkey]
    else:
        logging.info("I do not have a style for your label %s. "
                     "Returning standard."%label)
        return '-'
    
    
