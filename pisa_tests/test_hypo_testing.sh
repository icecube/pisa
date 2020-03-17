#!/bin/bash

#
# author: J.L. Lanfranchi
#
# Copyright (c) 2014-2020, The IceCube Collaboration
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


# TODO: following fails unless we can use larger data set size!
OUTDIR=$TMP/hypo_testing_test
echo "=============================================================================="
echo "Running hypo_testing.py, basic NMO Asimov analysis (not necessarily accurate)"
echo "Storing results to"
echo "  $OUTDIR"
echo "=============================================================================="
PISA_FTYPE=float32 python $PISA/pisa/scripts/analysis.py discrete_hypo \
	--h0-pipeline settings/pipeline/example.cfg \
	--h0-param-selections="ih" \
	--h1-param-selections="nh" \
	--data-param-selections="nh" \
	--data-is-mc \
	--min-settings settings/minimizer/l-bfgs-b_ftol2e-5_gtol1e-5_eps1e-4_maxiter200.json \
	--metric=chi2 \
	--logdir $OUTDIR \
	--pprint -v --allow-dirty
