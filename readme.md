# Chatbot - Summer Internship Project 2019 


### Project description
A generalised chatbot which can intelligently respond to conversation using named entity recognition (NER), sentiment analysis and an implementation of a decision system for follow up mechanism using a Seq2seq model which uses Bidirectional RNNs and Attention models.

### Encoder-Decoder Model
<!--Add the image from the project report folder-->

<!--create an images folder and change to relative path later

### Decision System for Follow Up
<!--Insert the flow chart here-->

### Installation
To run the chatbot program you must install the following libraries -
1. Install spacy 
``` deic
pip install spacy
```
2. Install sqlite3
```
pip install sqlite3
```
3. Install tensorflow
```
pip install tensorflow
```
4. Install dialogflow
```
pip install dialogflow
```
5. Install nltk
```
pip install nltk
```
6. Install tqdm
```
pip install tqdm
```
7. Install tensorlayer
```
pip install tensorlayer
```
8. Install wikipedia
```
pip install wikipedia
```
9. Install Watson NLU
```
pip install ibm_watson
```
10. Install Spacy's English Model
```
python -m spacy download en_core_web_md
```

### API's used -
1. MediaWiki
2. Dialogflow
3. IBM Watson NLU (for sentiment analysis)


### Training data
We have trained our model on three different datasets (SQuAD , Cornell Movie Dialogue Corpus and a Trump Tweets Dataset) to  bring about generalisation. They are included in the 'data' folder and are included through an import of the folder.


### Usage 
Note - First make sure that all the files required are saved in the same directory and all the required libraries are installed before running the program.
For Windows -
```
python model.py
```

For Linux (For Python3+ versions) -
```
python3 model.py
```
The model can be trained on the user's system or the pretrained weights (model.hdf5 files) included in the respective folders can be loaded to view the output.

### Results
The results of the chatbot (including the follow up) are shown below -
<!--Insert a screenshot of the results here-->

### References
<!--Insert the links of all blogs and resources that helped our cause-->
-----------------------------------------------------------------------------------------------------------------------------------
 
