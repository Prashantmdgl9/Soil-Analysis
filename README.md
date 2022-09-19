# Soil Analysis and Plant Disease Identification

During the covid lockdowns, my mother was buying a lot of plants online but many of them were dying within a couple of weeks. 
Now, neither of us was a botanist or a horticulurist(if that's a word), I looked up online to find if there is a database available that can show what plants leaves are diseased, so we can spot them in the early phase itself. 

The other thing I wanted to do was to identify the soil type automatically and provide suggestion that what should be grown in that type of soil. Presently, the model has been trained to identify 4 major types of soils viz. Alluvial, Black, Clay, and Red.

Eseentially, two tasks were:

1. Identify the soil type and plant only relevant seeds in it.

2. Identify any diseases early on and take corrective measure.

It was a simple mulit layered convolution network that I used and it yielded pretty decent results. There are of course massive room for improvement.

The app is present on free dynos of heroku 
https://soil-analyser.herokuapp.com/

and I published a blog on TowardsDataScience as well:
https://towardsdatascience.com/how-my-mother-started-using-cnn-for-her-plants-f0913e5548db
