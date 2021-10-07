# framegen
framegen generates frames of a video after training on it's frames.

# walkthrough
Clone this repository

`git clone` 

`cd framegen`


Set up virtualenv: 

`virtualenv venv`. 

Activate it with 

`source venv/bin/activate`.

Install requirements.

```pip install requirements.pip```

## preparing the frame data for training 
Download or move some video to  `./original-video.mp4`.

Scale it `./scale-vid.sh original-video.mp4 scaled-video.mp4 32`. This requires ffmpeg on your system. 32 is the y resolution of the outputted video. 

Create a dataset from the video: `python create-dataset-from-vid.py`.
You may want to edit the params in the script if you have deviated from this walkthrough guide.
This creates a folder `data` with all the frames from the video packed into either the training dataset or a smaller test dataset. 


## training 

Run the training `python train.py`.
You may want to edit the params in the script if you have deviated from this walkthrough guide.
This will create the model and write to `./models/...`.

The filename will depend on the parameters you have chosen for training like the number of epochs. 

During the training images will be saved to `./` that give an idea about the training progress.

![example training progress](/example-training-progress.png?raw=true "example training progress")

## seeing results

You can now try out the model. 
The script `play.py` will take a couple of frames from the test data set, encode frame by frame, interpolate lineary between the those frames, and use the model to decode these interpolated encoded frames creating a video that is written to `test-video.mp4`. 

Play sequential takes all the frames from the original video, encodes them frame by frame, and uses the decoder to decode them again, creating a video that get's written to `result-video.mp4`. 


