#!/bin/bash

#
# author: J.L. Lanfranchi
#
# Copyright (c) 2014-2017, The IceCube Collaboration
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License
#


BASEDIR=$(dirname "$0")
PISA=$BASEDIR/..
TMP=/tmp/pisa_tests
export PISA_RESOURCES=${TMP}/pisa_resources:$PISA_RESOURCES
mkdir -p $TMP
mkdir -p $PISA_RESOURCES
echo "PISA=$PISA"


echo "=============================================================================="
echo "Generating toy MC for use with test scripts"
echo "=============================================================================="
PISA_FTYPE=float32 python $PISA/pisa/scripts/make_toy_events.py --outdir ${PISA_RESOURCES}/events \
	--num-events 1e5 \
	--energy-range 1 80 \
	--spectral-index 1 \
	--coszen-range -1 1
echo "------------------------------------------------------------------------------"
echo "Finished creating toy MC events to be used with unit tests"
echo "------------------------------------------------------------------------------"
echo ""
echo ""


echo "=============================================================================="
echo "Running test_example_pipelines.py"
echo "=============================================================================="
python $BASEDIR/test_example_pipelines.py -v
echo "------------------------------------------------------------------------------"
echo "Finished Running test_example_pipelines.py"
echo "------------------------------------------------------------------------------"
echo ""
echo ""


# TODO: all files except setup.py and __init__.py that are listed below should
# have a command-line test defined further down in this script (i.e., these are
# scripts that require specific command-line arguments)
for f in `find $PISA/pisa -name "*.py"`
do
	BN=$(basename "$f")
	#if [[ "$BN" == test_* ]];then continue;fi
	if [[ "$f" == *pisa/scripts/* ]];then continue;fi
	if [ "$BN" == "__init__.py" ];then continue;fi
	if [ "$BN" == "setup.py" ];then continue;fi
	if [ "$BN" == pipeline.py ];then continue;fi
	if [ "$BN" == distribution_maker.py ];then continue;fi
	if [ "$BN" == genie.py ];then continue;fi

	echo "=============================================================================="
	echo "Running python $BN at abs path"
	echo "  `realpath $f`"
	echo "=============================================================================="
	python $f || FAILURE=true
	echo "------------------------------------------------------------------------------"
	echo "Finished running python $BN"
	echo "------------------------------------------------------------------------------"
	echo ""
	echo ""
	sleep 1
done


#
# Test CPU vs GPU, both FP64 and FP32 and CPU FP32 vs CPU FP64
#

OUTDIR_CPU64_NH_PIPELINE=$TMP/cpu64nh_pipeline
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with CPU & fp64 selected."
echo "Storing results to"
echo "  $OUTDIR_CPU64_NH_PIPELINE"
echo "=============================================================================="
PISA_FTYPE=float64 python $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	--select "nh" \
	--outdir $OUTDIR_CPU64_NH_PIPELINE \
	--png -v

OUTDIR_CPU32_NH_PIPELINE=$TMP/cpu32nh_pipeline
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with CPU & fp32 selected."
echo "Storing results to"
echo "  $OUTDIR_CPU32_NH_PIPELINE"
echo "=============================================================================="
PISA_FTYPE=float32 python $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	-a stage.aeff param.aeff_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	-a stage.reco param.reco_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	--select "nh" \
	--outdir $OUTDIR_CPU32_NH_PIPELINE \
	--png -v

OUTDIR_GPU64_NH_PIPELINE=$TMP/gpu64nh_pipeline
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with GPU & fp64 selected."
echo "Storing results to"
echo "  $OUTDIR_GPU64_NH_PIPELINE"
echo "=============================================================================="
PISA_FTYPE=float64 python $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	-a stage.aeff param.aeff_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	-a stage.reco param.reco_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	--select "nh" \
	--outdir $OUTDIR_GPU64_NH_PIPELINE \
	--png -v

OUTDIR_GPU32_NH_PIPELINE=$TMP/gpu32nh_pipeline
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with GPU & fp32 selected."
echo "Storing results to"
echo "  $OUTDIR_GPU32_NH_PIPELINE"
echo "=============================================================================="
PISA_FTYPE=float32 python $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	-a stage.aeff param.aeff_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	-a stage.reco param.reco_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	--select "nh" \
	--outdir $OUTDIR_GPU32_NH_PIPELINE \
	--png -v

OUTDIR=$TMP/compare_cpu64nh_pipeline_gpu64nh_pipeline
echo "=============================================================================="
echo "Running compare.py, CPU vs. GPU pipeline settings, FP64."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
PISA_FTYPE=float64 python $PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_CPU64_NH_PIPELINE/*.json* \
	--ref-label 'cpu64nh' \
	--test $OUTDIR_GPU64_NH_PIPELINE/*.json* \
	--test-label 'gpu64nh' \
	--outdir $OUTDIR \
	--png -v

OUTDIR=$TMP/compare_cpu32nh_pipeline_gpu32nh_pipeline
echo "=============================================================================="
echo "Running compare.py, CPU vs. GPU pipeline settings, FP32."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
PISA_FTYPE=float32 python $PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_CPU32_NH_PIPELINE/*.json* \
	--ref-label 'cpu32nh' \
	--test $OUTDIR_GPU32_NH_PIPELINE/*.json* \
	--test-label 'gpu32nh' \
	--outdir $OUTDIR \
	--png -v

OUTDIR=$TMP/compare_cpu32nh_pipeline_cpu64nh_pipeline
echo "=============================================================================="
echo "Running compare.py, CPU32NH vs. CPU64NH"
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
PISA_FTYPE=float64 python $PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_CPU64_NH_PIPELINE/*.json* \
	--ref-label 'cpu64nh' \
	--test $OUTDIR_CPU32_NH_PIPELINE/*.json* \
	--test-label 'cpu32nh' \
	--outdir $OUTDIR \
	--png -v


#
# Test hierarchy NH vs IH
#

OUTDIR_CPU64_IH_PIPELINE=$TMP/cpu64ih_pipeline
echo "=============================================================================="
echo "Running pipeline.py with example.cfg, with ih selected."
echo "Storing results to"
echo "  $OUTDIR_CPU64_IH_PIPELINE"
echo "=============================================================================="
PISA_FTYPE=float64 python $PISA/pisa/core/pipeline.py \
	-p settings/pipeline/example.cfg \
	-a stage.aeff param.aeff_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	-a stage.reco param.reco_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
	--select "ih" \
	--outdir $OUTDIR_CPU64_IH_PIPELINE \
	--png -v

OUTDIR=$TMP/compare_cpu64nh_pipeline_to_cpu64ih_pipeline
echo "=============================================================================="
echo "Running compare.py, nh vs. ih MapSets produced above with plots."
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
python $PISA/pisa/scripts/compare.py \
	--ref $OUTDIR_CPU64_IH_PIPELINE/*.json* \
	--ref-label 'cpu64ih' \
	--test $OUTDIR_CPU64_NH_PIPELINE/*.json* \
	--test-label 'cpu64nh' \
	--outdir $OUTDIR \
	--png -v


#
# Test that DistributionMaker has same result as pipeline
#

# TODO: removed since -a option doesn't work for distmaker
#OUTDIR_CPU64_NH_DISTMAKER=$TMP/cpu64nh_distmaker
#echo "=============================================================================="
#echo "Running distribution_maker.py with example.cfg, with nh selected."
#echo "Storing results to"
#echo "  $OUTDIR_CPU64_NH_DISTMAKER"
#echo "=============================================================================="
#PISA_FTYPE=float64 python $PISA/pisa/core/distribution_maker.py \
#	-p settings/pipeline/example.cfg \
#	-a stage.aeff param.aeff_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
#	-a stage.reco param.reco_events=events/events__vlvnt__toy_1_to_80GeV_spidx1.0_cz-1_to_1_1e5evts_set0__unjoined.hdf5 \
#	--select "nh" \
#	--outdir $OUTDIR_CPU64_NH_DISTMAKER \
#	--png -v
#
#OUTDIR=$TMP/compare_cpu64nh_distmaker_to_cpu64nh_pipeline
#echo "=============================================================================="
#echo "Running compare.py, fp64/cpu distmaker vs. fp64/cpu pipeline-produced MapSets."
#echo "Storing results to"
#echo "  $OUTDIR"
#echo "=============================================================================="
#python $PISA/pisa/scripts/compare.py \
#	--ref $OUTDIR_CPU64_NH_PIPELINE/*.json* \
#	--ref-label 'cpu64nh_pipeline' \
#	--test $OUTDIR_CPU64_NH_DISTMAKER/*.json* \
#	--test-label 'cpu64nh_distmaker' \
#	--outdir $OUTDIR \
#	--png -v


# Call script to run hypothesis testing (runs minimizer with a pipeline)
$BASEDIR/test_hypo_testing.sh
