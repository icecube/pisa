# pisa.stages

Directories are PISA stages, and within each directory can be found the services implementing the respective stage.

# Anatomy of a Stage

The PISA stage inherits from pisa.core.stage.Stage

There are 4 pasrts to a stage:

## The constructor

Assign here any arguments passed in via the config, define expected parameters, and init the base stage

## The setup function

## The calculation function

## The apply function
