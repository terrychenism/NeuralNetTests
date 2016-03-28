#!/bin/bash

EXPECTED_ARGS=2
E_BADARGS=65

# if [ $# -lt $EXPECTED_ARGS ]
# then
#   echo "Usage: `basename $0` video frames/sec [size=256]"
#   exit $E_BADARGS
# fi



for f in UCF101/*.avi; 
do 
	# echo "Processing $f ";
	NAME=${f%.*}
	FRAMES=30
	BNAME=`basename $NAME`
	echo $BNAME
	mkdir frames/$BNAME

	ffmpeg.exe -i $f -r $FRAMES frames/$BNAME/$BNAME.%4d.jpg
	
done
