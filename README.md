# face-recognition-exploration

Evaluates the precision & recall of AWS Rekognition on the LFW dataset.

To run:
1)  Setup a Python 3.6 virtual environment and install the requirements:
    $ pip - r requirements.txt

2)  Download and unzip the LFW images into a folder called 'input_images':
    http://vis-www.cs.umass.edu/lfw/

3)  Put your AWS credentials into process.py

4)  $ python process.py

TODOS:

1)  k-fold cross-validation
2)  plotting results with plot.ly
