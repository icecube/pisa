# Stage: data

This folder contains various stages designed to either load existing data, or create fake data events via some Toy Mont Carlo generators

## Services

Several services are available, though only `simple_data_loader` is used an regularly maintained in the context of oscillation analyses:

### simple_data_loader

This is the main service used by all oscNext oscillation analyses. Loads pisa-compatible hdf5 files containing harvested variables from the event selection i3 files.

### toy_event_generator

This service creates a set of toy MC neutrino events, based on an arbitrary choice of flux. This fake data is binned using the same convention as the normal oscillation analysis (ie, events are binned in energy, coszen and PID)

### simple

This service creates a simple, 1D dataset consisting of a gaussian signal on top of a uniform background. Useful to make simple checks on minimization and likelihood implementations. `super_simple_pipeline.cfg` provides an example pipeline that ca be used to run this service

### csv_data_hist

Loading data from iceCube datareleases

### csv_icc_hist

Loading muons from iceCube datareleases

### csv_loader

Loading MC from iceCube datareleases

### grid

Description missing
