## Mask detection
It is a python application that uses [TensorFlow](https://www.tensorflow.org/) and [OpenCV](https://opencv.org/) for real-time face mask detection.

## Resources
The 3rd-party resources are already integrated with the application. This is a reference list to find the original artfacts:
- Face detection model and the script to download its weights can be found [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector).
- The dataset of people wearing masks can be found [here](https://github.com/prajnasb/observations/tree/master/experiements/data).
- The tutorial that inspired this project can be found [here](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/).

# Libraries
This project make use of three main libraries:
- __OpenCV:__ to run the video stream and detect when there is one or more faces in the frame;
- __imutils:__ to handle image (the video frames) manipulations;
- __Tensorflow 2.0:__ to create, train and predict if the person in the video frame is wearing a mask.
