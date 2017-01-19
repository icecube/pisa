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
    pretty_labels = {}
    pretty_labels["atm_muon_scale"] = r"Muon Background Scale"
    pretty_labels["nue_numu_ratio"] = r"$\nu_e/\nu_{\mu}$ Ratio"
    pretty_labels["nu_nubar_ratio"] = r"$\nu/\bar{\nu}$ Ratio"
    pretty_labels["Barr_uphor_ratio"] = r"Barr Up/Horizontal Ratio"
    pretty_labels["barr_uphor_ratio"] = r"Barr Up/Horizontal Ratio"
    pretty_labels["Barr_nu_nubar_ratio"] = r"Barr $\nu/\bar{\nu}$ Ratio"
    pretty_labels["barr_nu_nubar_ratio"] = r"Barr $\nu/\bar{\nu}$ Ratio"
    pretty_labels["delta_index"] = r"Atmospheric Index Change"
    pretty_labels["theta13"] = r"$\theta_{13}$"
    pretty_labels["theta23"] = r"$\theta_{23}$"
    pretty_labels["sin2theta23"] = r"$\sin^2\theta_{23}$"
    pretty_labels["deltam31"] = r"$\Delta m^2_{31}$"
    pretty_labels["aeff_scale"] = r"$A_{\mathrm{eff}}$ Scale"
    pretty_labels["energy_scale"] = r"Energy Scale"
    pretty_labels["Genie_Ma_QE"] = r"GENIE $M_{A}^{QE}$"
    pretty_labels["genie_ma_qe"] = r"GENIE $M_{A}^{QE}$"
    pretty_labels["Genie_Ma_RES"] = r"GENIE $M_{A}^{Res}$"
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
    summary file in the logdir. This should account for dimensions though is 
    only tested with degrees.
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
        parse_string = ('gaussian prior: stddev=(.*) (.*)'
                        ', maximum at (.*) (.*)')
        bits = re.match(
            parse_string,
            prior_string,
            re.M|re.I
        )
        stddev = float(bits.group(1))
        maximum = float(bits.group(3))

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
