# HackathonProj
![Figure_1](https://user-images.githubusercontent.com/98052052/217681302-f1ecb925-ea8f-4f32-bddb-19e48aa14406.png)
## Inspiration
"Not A Hotdog" from the movie "Silicon Valley" as a joke. However, AI is intriguing so we decided to move forward with our (semi-complete) version of an image recognition system.
## What it does
When run, the model trains on a dataset of over 60,000 fashion piece images. After, the model randomly selects 16 images from a separate fashion dataset (NEW to the model) and predicts the category the piece falls under. The confidence percentages are displayed in a bar graph, sometimes showing that the model predicted incorrectly or had a few options in mind.

Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot

## How we built it
Using Python as our primary, we were able to utilize TensorFlow's library in order to load in our data sets and train them. From there, we were able to test and predict using more of TensorFlow's functions, and, finally, used matplotlib to format our output and provide meaningful numbers.

## Challenges we ran into
AI, deep learning, and computer vision models are a lot more complex than we thought. There's a lot to that realm of CS and we realized pretty quickly that a full-blown, personal, configurable app may be out of the 24 hour window. Figuring out how to train different datasets as well as configuring good training and test data was by far the most challenging part. Fortunately, our model reached 93% confidence, but that was due to the uniformity and depth of the Fashion MNIST dataset.

## Accomplishments that we're proud of
We had no prior knowledge of deep learning, AI, or computer vision models. So just learning and beginning to understand those topics was a huge accomplishment to both of us.

## What we learned
Sometimes you have to just dive in and ask questions later. We spent the first four hours really trying to understand some complex topics, and although we may not have developed a robust, ground-breaking model, we stuck with it.

## What's next for Computer Vision Model - Fashion
It would be awesome to be able to compare single images with our model. This is a feature we worked really hard to try and figure out, but never got the compatibility ironed out. This would allow users to take a picture of an article of clothing and have our model predict what category of clothing it is. As we said, this was our ultimate goal, we just didn't get to it. Although, we still think our model is pretty cool.
