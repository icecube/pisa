
'''
Convert GRECO event sample from .i3 files to PISA event files
Tom Stuttard
'''

#TODO Enforce correct environment (e.g. IceCube, not PISA)


if __name__ == "__main__" :

	import glob, datetime

	start_time = datetime.datetime.now()

	#from pisa.utils.i3_to_pisa import I3ToPISAVariableMap, convert_i3_to_pisa
	from i3_to_pisa import I3ToPISAVariableMap, convert_i3_to_pisa

	debug = True


	#
	# Define input data
	#

	#TODO Do we need an "expected_variables" member in PISA stages?

	#TODO Need to do data, nominal sim, systematics sets, etc

	#TODO Use latest GRECO, but need to get correct pegleg variable names...

	#Define inputs files for each category of events
	input_data = {}

	#IceCube experimental data (pass 1)
	input_data["exp_pass1"] = sorted(glob.glob("/data/ana/LE/NBI_nutau_appearance/level7_24Nov2015/exp/20??/Level7_data_IC2014.Run??????.i3.bz2"))

	#Truncate input data if debugging
	if debug :
		for k in input_data.keys() :
			input_data[k] = input_data[k][:10]


	#
	# Define mapping of variables from i3 to PISA
	#

	#Map variables
	variable_map = I3ToPISAVariableMap()
	variable_map.add("Pegleg_Fit_MNHDCasc","energy","reco_energy_cascade")
	variable_map.add("Pegleg_Fit_MNTrack","energy","reco_energy_track")


	#
	# Run the conversion
	#

	output_file = "greco.hdf5"

	#Call the main i3 -> PISA converter function to do the heavy lifting
	convert_i3_to_pisa(input_data=input_data,
						output_file=output_file,
						variable_map=variable_map) 



	#
	# Clean up
	#

	#TODO Combine pegleg variables


	#
	# Done
	#

	end_time = datetime.datetime.now()

	print "Done! Took %s" % (end_time-start_time)



