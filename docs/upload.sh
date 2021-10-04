#!/bin/sh

LOCALDIR=$(dirname "$0")
HOST=pub2.icecube.wisc.edu
HTML_DIR=~peller/public_html/pisa_docs

echo "Creating remote HTML directory if it doesn't already exist"
ssh $HOST "test -d $HTML_DIR/ || mkdir $HTML_DIR/"

echo "Removing contents of remote HTML folder"
ssh $HOST "find $HTML_DIR/ -mindepth 1 -print0 | xargs -0 rm -rf"

echo "Uploading new documentation"
scp -r $LOCALDIR/build/html/* $HOST:$HTML_DIR/

echo "Changing permissions on uploaded files"
ssh $HOST "find $HTML_DIR/ -mindepth 1 -print0 | xargs -0 chmod a=u"

#RSYNC=/usr/bin/rsync
#$RSYNC \
#	--protocol=30 \
#	--delete --force \
#	--recursive \
#	--compress --compress-level=9 \
#	--human-readable --progress --stats \
#	-vvv \
#	$THISDIR/build/html/ $HOST:$HTML_DIR/ \
#	&& \
#	ssh $HOST "find $HTML_DIR/ -print0 | xargs -0 chmod a=u"
