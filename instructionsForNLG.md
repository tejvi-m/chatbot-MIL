clone the repo ```NLG-DAE```

# preprocessing the dataset
any dataset with Questions and ANswers will do.
for now we will only be using the questions.
have a csv file that only contains the questions under the column "ref"
for every question, go through the whole dataset and remove all questions that contain dates in them,
and you can do this to all teh questions that have proper nouns in them too but im guessing that would affect the training set size significantly.
run the script throught ```newCorrupt.py```
you should have a new csv file that has another column called "corrupted"

#running the model
run ```trainNER.py```. the input pipeline might be messed up (that is the part where you read the csv file). Take into account the corrections ive mailed you, and if there's still any probkem with the feeding in part, then fix it.
change the save every to 100 or something if youre training it locally, but if youre using colab let it remain at a higher number.

#running inference
change the List ```L``` and run ```inferenceNER.py```
