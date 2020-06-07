Sample Trained Discriminator Model: https://www.dropbox.com/sh/c2xpka0n3ro09j8/AADvpqw817mouInopOsfbIyIa?dl=0
Sample Trained Generator Model: https://www.dropbox.com/sh/v1cl58y5whd67y7/AACjSaCtypMv9bCQoC-sblAPa?dl=0

Sample Dataset: https://www.dropbox.com/sh/ux7b3n9ra0f4hpc/AABoFEoxzzewtwV1QnDOVuRTa?dl=0

To run the the program the python comman needs to be usabel in the terminal/command prompt.

run the following command in the terminal/command prompt:
    python filepathto/HRGAN_Controller.py numOfChunks
where filepathto is the path to HRGAN_Controller.py and numOfChunks is
(fullIMGXRes**2)/(chunkIMGXRes**2) eg. (512**2)/(256**2) = 4 chunks