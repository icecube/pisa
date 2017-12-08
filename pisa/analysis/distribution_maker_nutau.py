from pisa.core.distribution_maker import DistributionMaker 
import numpy as np
import copy

class DistributionMakerNutau(DistributionMaker):

    def get_variables(self, variables, no_reco=False, apply_sys_to_mc=True, muon_cuts='icc_def2', data_cuts='analysis', muon_proc='events/data_proc_params.json', data_proc='events/data_proc_params.json', return_stages=['mc','icc'], pid_selection=''):
        # Note: only works with mc, icc (or corsika) and/or data stages
        print "muon_cuts= ", muon_cuts, ", data_cuts=", data_cuts 
        print "muon_proc_file = ", muon_proc
        print "data_proc_file = ", data_proc
        pipelines = self._pipelines
        mc_stage = None
        combine_stage = None
        sys_stage = None
        muon_stage = None
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
                if pipelines[i]['data'].service_name == 'icc' or pipelines[i]['data'].service_name == 'corsika':
                    muon_stage = pipelines[i]['data'] 
                if pipelines[i]['data'].service_name == 'data':
                    data_stage = pipelines[i]['data'] 

        if pid_selection!='':
            assert(pid_selection in ['cscd', 'trck'])
            if 'pid' not in variables:
                variables.append('pid')

        # params for data
        if data_stage is not None and 'data' in return_stages:
            #data_params = data_stage.get_fields(fields=variables, no_reco=no_reco, cuts=data_cuts, data_proc_file=data_proc)
            data_params = data_stage.get_fields(fields=variables, no_reco=no_reco, cuts=data_cuts)
        else:
            data_params = None

        # params for icc 
        if muon_stage is not None:
            if 'icc' in return_stages:
                muon_params = muon_stage.get_fields(fields=variables, no_reco=no_reco, cuts = muon_cuts, event_file = muon_stage.params.icc_bg_file.value, data_proc_file=muon_proc)
            if 'corsika' in return_stages:
                muon_params = muon_stage.get_fields(fields=variables, no_reco=no_reco, cuts = muon_cuts, event_file = muon_stage.params.corsika_file.value, data_proc_file=muon_proc)
        else:
            muon_params = None

        # params for mc 
        if mc_stage is not None and 'mc' in return_stages:
            mc_variables=copy.deepcopy(variables)
            if no_reco==False:
                # weight and sumw2 will be used in plotting script, 'reco_energy', 'reco_coszen', 'pid' are the binning 
                mc_variables_add=['weight', 'sumw2', 'reco_energy', 'reco_coszen', 'pid']
            else:
                mc_variables_add=['weight', 'sumw2']
            #for param in ['weight', 'sumw2', 'reco_energy', 'reco_coszen', 'pid']:
            for param in mc_variables_add:
                if param not in mc_variables:
                    mc_variables.append(param)
            #print "in distribution_maker_nutau, mc_variables ", mc_variables
            mc_params = mc_stage.get_fields(fields=mc_variables, no_reco=no_reco)
            if apply_sys_to_mc:
                if combine_stage is not None:
                    transforms = combine_stage.get_transforms()
                    print "applying transforms in combine_stage"
                    for i in range(0, len(transforms)):
                        transform = transforms[i]
                        input_names = transform.input_names
                        output_name = transform.output_name
                        transform_array = transform.xform_array
                        for idx,flav in enumerate(input_names):
                            bin_idx=[]
                            for idx2, (bin_name, bin_edges) in enumerate(zip(transform.output_binning.names, transform.output_binning.bin_edges)):
                                digitized_idx = np.digitize(mc_params[flav][bin_name], bin_edges)
                                # get the index starting from 0
                                digitized_idx -= 1
                                bin_idx.append(digitized_idx)
                            transform_in_bin = np.array([transform_array[idx][i][j][k] for i,j,k in zip(bin_idx[0], bin_idx[1], bin_idx[2])])
                            mc_params[flav]['weight']*= transform_in_bin
                if sys_stage is not None:
                    transforms = sys_stage.get_transforms()
                    print "applying transforms in sys_stage"
                    combine_groups = combine_stage.combine_groups
                    for i in range(0, len(transforms)):
                        transform = transforms[i]
                        input_names = transform.input_names
                        output_name = transform.output_name
                        transform_array = transform.xform_array
                        for idx, input_flav in enumerate(input_names):
                            flavs_to_apply = combine_groups[input_flav]
                            for flav in flavs_to_apply:
                                bin_idx = []
                                for idx2, (bin_name, bin_edges) in enumerate(zip(transform.output_binning.names, transform.output_binning.bin_edges)):
                                    digitized_idx = np.digitize(mc_params[flav][bin_name], bin_edges)
                                    # get the index starting from 0
                                    digitized_idx -= 1
                                    bin_idx.append(digitized_idx)
                                transform_in_bin = np.array([transform_array[i][j][k] for i,j,k in zip(bin_idx[0], bin_idx[1], bin_idx[2])])
                                mc_params[flav]['weight']*= transform_in_bin
        else:
            mc_params = None
        tol=0

        if pid_selection!='':
            if data_params is not None:
                pid_copy = copy.deepcopy(data_params['pid'])
                if pid_selection=='cscd':
                    data_cut = np.logical_and(pid_copy>=-3, pid_copy<2.0)
                if pid_selection=='trck':
                    data_cut = pid_copy>=2.0
                for key in data_params.keys():
                    data_params[key] = data_params[key][data_cut]
            if muon_params is not None:
                pid_copy = copy.deepcopy(muon_params['pid'])
                if pid_selection=='cscd':
                    pid_cut = np.logical_and(pid_copy>=-3, pid_copy<2.0)
                if pid_selection=='trck':
                    pid_cut = pid_copy>=2.0
                for key in muon_params.keys():
                    muon_params[key] = muon_params[key][pid_cut]
            if mc_params is not None:
                for flav in mc_params.keys():
                    pid_copy = copy.deepcopy(mc_params[flav]['pid'])
                    if pid_selection=='cscd':
                        mc_cut = np.logical_and(pid_copy>=-3, pid_copy<2.0)
                    if pid_selection=='trck':
                        mc_cut = pid_copy>=2.0
                    for key in mc_params[flav].keys():
                        mc_params[flav][key] = mc_params[flav][key][mc_cut]
        return mc_params, muon_params, data_params
