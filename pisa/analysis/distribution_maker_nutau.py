from pisa.core.distribution_maker import DistributionMaker 
import numpy as np
import copy

class DistributionMakerNutau(DistributionMaker):

    def get_variables(self, variables, apply_sys_to_mc=True, return_stages=['mc','icc'], pid_selection=''):
        # Note: only works with mc, icc and/or data stages

        #print "apply_sys_to_mc ", apply_sys_to_mc
        pipelines = self._pipelines
        mc_stage = None
        combine_stage = None
        sys_stage = None
        icc_stage = None
        data_stage = None
        for i in range(0, len(pipelines)):
            pipe = pipelines[i]
            #print "pipe stages = ", pipe.stage_names
            pipeline_stage_names = pipelines[i].stage_names
            if 'mc' in pipeline_stage_names:
                mc_stage = pipelines[i]['mc']
            if 'combine' in pipeline_stage_names:
                combine_stage = pipelines[i]['combine']
            if 'discr_sys' in pipeline_stage_names:
                sys_stage = pipelines[i]['discr_sys']
            if 'data' in pipeline_stage_names:
                if pipelines[i]['data'].service_name == 'icc':
                    icc_stage = pipelines[i]['data'] 
                if pipelines[i]['data'].service_name == 'data':
                    data_stage = pipelines[i]['data'] 

        if pid_selection!='':
            assert(pid_selection in ['cscd', 'trck'])
            if 'pid' not in variables:
                variables.append('pid')

        # params for data
        if data_stage != None and 'data' in return_stages:
            data_params = data_stage.get_fields(fields=variables)
        else:
            data_params = None

        # params for icc 
        if icc_stage != None and 'icc' in return_stages:
            icc_params = icc_stage.get_fields(fields=variables, icc_file_name = icc_stage.params.icc_bg_file.value)
        else:
            icc_params = None

        # params for mc 
        if mc_stage != None and 'mc' in return_stages:
            mc_variables=copy.deepcopy(variables)
            for param in ['weight', 'sumw2', 'reco_energy', 'reco_coszen', 'pid']:
                # weight and sumw2 will be used in plotting script, 'reco_energy', 'reco_coszen', 'pid' are the binning 
                if param not in mc_variables:
                    mc_variables.append(param)
            mc_params = mc_stage.get_fields(fields=mc_variables)
            if apply_sys_to_mc:
                if combine_stage !=None:
                    transforms = combine_stage.get_transforms()
                    for i in range(0, len(transforms)):
                        transform = transforms[i]
                        input_names = transform.input_names
                        output_name = transform.output_name
                        transform_array = transform.xform_array
                        #print "shape transform_array", np.shape(transform_array)
                        #print "input_names", input_names
                        for idx,flav in enumerate(input_names):
                            bin_idx=[]
                            for idx2, (bin_name, bin_edges) in enumerate(zip(transform.output_binning.names, transform.output_binning.bin_edges)):
                                digitized_idx = np.digitize(mc_params[flav][bin_name], bin_edges)
                                # get the index starting from 0
                                digitized_idx -= 1
                                bin_idx.append(digitized_idx)
                            transform_in_bin = np.array([transform_array[idx][i][j][k] for i,j,k in zip(bin_idx[0], bin_idx[1], bin_idx[2])])
                            mc_params[flav]['weight']*= transform_in_bin
                if sys_stage != None:
                    transforms = sys_stage.get_transforms()
                    #print "In sys stage, len transforms", len(transforms)
                    for i in range(0, len(transforms)):
                        transform = transforms[i]
                        input_names = transform.input_names
                        output_name = transform.output_name
                        transform_array = transform.xform_array
                        #print "     shape transform_array", np.shape(transform_array)
                        #print "     sys in input_names: ", input_names
                        #print "     sys in output_names: ", output_name
                        for idx, input_flav in enumerate(input_names):
                            #print "idx, input_flav", idx, " ", input_flav
                            if 'nc' in input_flav:
                                flavs_to_apply=['numu_nc', 'nutau_nc', 'nue_nc', 'numubar_nc', 'nutaubar_nc', 'nuebar_nc']
                            if input_flav=='nue_cc':
                                flavs_to_apply=['nue_cc', 'nuebar_cc']
                            if input_flav=='numu_cc':
                                flavs_to_apply=['numu_cc', 'numubar_cc']
                            if input_flav=='nutau_cc':
                                flavs_to_apply=['nutau_cc', 'nutaubar_cc']
                            for flav in flavs_to_apply:
                                bin_idx = []
                                for idx2, (bin_name, bin_edges) in enumerate(zip(transform.output_binning.names, transform.output_binning.bin_edges)):
                                    digitized_idx = np.digitize(mc_params[flav][bin_name], bin_edges)
                                    # get the index starting from 0
                                    digitized_idx -= 1
                                    bin_idx.append(digitized_idx)
                                transform_in_bin = np.array([transform_array[i][j][k] for i,j,k in zip(bin_idx[0], bin_idx[1], bin_idx[2])])
                                #print "len transform_in_bin", len(transform_in_bin)
                                mc_params[flav]['weight']*= transform_in_bin
        else:
            mc_params = None

        if pid_selection!='':
            if data_params!=None:
                pid_copy = copy.deepcopy(data_params['pid'])
                if pid_selection=='cscd':
                    data_cut = np.logical_and(pid_copy>=-3, pid_copy<2.0)
                if pid_selection=='trck':
                    data_cut = pid_copy>=2.0
                for key in data_params.keys():
                    data_params[key] = data_params[key][data_cut]
            if icc_params!=None:
                pid_copy = copy.deepcopy(icc_params['pid'])
                if pid_selection=='cscd':
                    icc_cut = np.logical_and(pid_copy>=-3, pid_copy<2.0)
                if pid_selection=='trck':
                    icc_cut = pid_copy>=2.0
                for key in icc_params.keys():
                    if key=='weight':
                        continue
                    icc_params[key] = icc_params[key][icc_cut]
            if mc_params!=None:
                for flav in mc_params.keys():
                    pid_copy = copy.deepcopy(mc_params[flav]['pid'])
                    if pid_selection=='cscd':
                        mc_cut = np.logical_and(pid_copy>=-3, pid_copy<2.0)
                    if pid_selection=='trck':
                        mc_cut = pid_copy>=2.0
                    for key in mc_params[flav].keys():
                        mc_params[flav][key] = mc_params[flav][key][mc_cut]

        return mc_params, icc_params, data_params
