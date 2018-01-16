import glob, sys, os, tables

from icecube import icetray
from I3Tray import I3Tray
from icecube.tableio import I3TableWriter, I3TableService
from icecube.hdfwriter import I3HDFTableService #, pytables_ext


#
# Handle variable mapping
#

class VariableMap() :

	class Variable() :

		def __init__(self,i3_frame_object,i3_frame_object_variable,pisa_variable) :
			self.i3_frame_object = i3_frame_object
			self.i3_frame_object_variable = i3_frame_object_variable
			self.pisa_variable = pisa_variable

	def __init__(self) :
		self.variables = []


	def add(self,i3_frame_object,i3_frame_object_variable,pisa_variable) :
		#TODO Check no duplication in pisa_variable
		self.variables.append( self.Variable(i3_frame_object,i3_frame_object_variable,pisa_variable) )


	def get_all_i3_frame_objects(self) :
		return list(set([ v.i3_frame_object for v in self.variables ]))

	def get_all_pisa_variables(self) :
		return [ v.pisa_variable for v in self.variables ]

	def __getitem__(self,i3_frame_obj) :
		return [ v for v in self.variables if v.i3_frame_object == i3_frame_obj ]




#
# Helper functions
#

def i3_to_icecube_hdf5(i3_files,variable_map,print_frame_objects=False) :

	output_file = "tmp.hdf5"

	i3_variables = variable_map.get_all_i3_frame_objects()

	tray = I3Tray()

	hdf = I3HDFTableService(output_file)

	tray.AddModule('I3Reader','reader',filenamelist=i3_files)

	if print_frame_objects :
		def print_frame(frame) : print frame
		tray.AddModule(print_frame)

	#Check all required i3 frame objects exist
	def check_all_frame_objects_exist(frame) :
		for var in i3_variables :
			if var not in frame :
				return False
		return True
	tray.AddModule(check_all_frame_objects_exist,"check_frame_objects")

	tray.AddModule(I3TableWriter,'writer',
	               tableservice = hdf,
	               #tableservice=my_table,
	               SubEventStreams = ["InIceSplit"], #TODO Is this the one?
	               #keys         = ['Pegleg_Fit_MNHDCasc','Pegleg_Fit_MNTrack']
	               #keys         = ['SRTTWOfflinePulsesDC']
	               #keys         = ['TauL6_bool','SRTTWOfflinePulsesDC']
	               keys = i3_variables
	              )

	tray.Execute()

	return output_file





#
# Define inputs
#

#Input files
input_files = {}
#input_files["data"] = sorted(glob.glob("/data/ana/LE/NBI_nutau_appearance/level7_5July2017/data/2014/Level7_Run??????.*.i3.bz2"))
#input_files["data"] = sorted(glob.glob("/data/ana/LE/NBI_nutau_appearance/level7_5July2017/data/2014/Level7_Run??????.0001.i3.bz2"))
input_files["data"] = sorted(glob.glob("/data/ana/LE/NBI_nutau_appearance/level7_24Nov2015/exp/2014/Level7_data_IC2014.Run??????.i3.bz2"))
#TODO Use latest GRECO, but need to get correct pegleg variable names...

#Map variables
variable_map = VariableMap()
variable_map.add("Pegleg_Fit_MNHDCasc","energy","true_energy_cascade")
variable_map.add("Pegleg_Fit_MNTrack","energy","true_energy_track")



#
# Create a top-level data system
#

#TODO Create a PISA events file class?

#Create the file
output_pisa_events_file = tables.open_file("pisa_events.hdf5", mode="w", title="PISA HDF5 Events File")

#Create an events group
events_group = output_pisa_events_file.create_group("/", 'Events', 'Event data')

#TODO Create metadata



#
# Loop over datasets
#

i3_variables = variable_map.get_all_i3_frame_objects()

for dataset_key,files in input_files.items() :

	#TODO Handle CC,NC, etc


	#
	# Convert i3 files to IceCube HDF5 format
	#

	print "\n%s : %i files" % (dataset_key,len(files))

	files = files[:10] #TODO REMOVE

	tmp_hdf5_file_path = i3_to_icecube_hdf5(files,variable_map,print_frame_objects=False)



	#
	# Add the data to the top-level output HDF5 file
	#

	#Load the temporary i3 file
	tmp_hdf5_file = tables.openFile(tmp_hdf5_file_path)

	#Create a group for this dataset_key in the output file
	dataset_key_group = output_pisa_events_file.create_group(events_group, dataset_key, dataset_key)

	#Loop over variables
	for i3_var in i3_variables :

		#Get the table correspondig to this variable in tmp hdf5 file
		i3_var_table = getattr(tmp_hdf5_file.root,i3_var)

		#Get all mappings for this i3 variable
		mappings = variable_map[i3_var]

		#Loop through mappings
		for mapping in mappings :

			#Check frame object variable is present #TODO DO this earlier???
			if mapping.i3_frame_object_variable not in i3_var_table.colnames :
				raise Exception("Error : Frame object %s does not contain the variable %s" % (i3_var,mapping.i3_frame_object_variable) ) 

			#Get the variable and add it to the output PISA HDF5 file group
			print i3_var_table.col(mapping.i3_frame_object_variable)[:5]
			#pisa_variable

			#Get the variable and add it to the output PISA HDF5 file group
			#Need to create the array if this is the first time we've tried to add daat to it
			i3_var_array = i3_var_table.col(mapping.i3_frame_object_variable)
			if mapping.pisa_variable in dataset_key_group :
				pisa_var_array = get_attr(dataset_key_group, mapping.pisa_variable)
				i3_var_array.append(i3_var_array)
			else :
				output_pisa_events_file.create_array(dataset_key_group, mapping.pisa_variable, i3_var_array, "") #TODO Add description field (optional in VariableMapping)?? 
				#output_pisa_events_file.create_carray(dataset_key_group, mapping.pisa_variable, i3_var_array, "") #TODO array or carray?


	#TODO NC/CC

	#output_data[dataset_key] = from_file(tmp_file)

	#TODO Free memory?

	#TODO Delete tmp files?

	#TODO Check all pisa_variables have had array created, and that all have same number of elements


#
# Write the top-level file
#

print "\nOutput file:"
print output_pisa_events_file

#output_file = "output.hdf5"
#to_file(output_data[dataset_key],output_file)

print "Done!"




