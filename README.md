# Optical Character Recognition(OCR)
Computational Robotics 2015 Fall Computer Vision Project.

We've built a ROS Package that uses Neato's camera to detect signs such as digits from 0 to 9 and alphabet letters and predict what the digit or letter is.

[Here](http://) You can find a video of our project.

## System 


## Design
A design decision we had to make was how to deal with noise. The model will predict a digit for whatever the Neato sees, even if there is not a sign present, so we needed to filter the predictions in some way. The first thing that we did was check the probability that the model assigned to its prediction and only accept it if it was over 50% confident. We also added a sliding window of 5 frames and only used the predictions if they all agreed for those frames. Together, these measures were very successful at making it so that the Neato would only act when it was actually reading a sign.

## Software Architecture


## Challenges




## Future Work

## Lessons
