# Chatbot


### Project description
A generalised chatbot which can intelligently respond to conversation using named entity recognition (NER), sentiment analysis and an implementation of a basic folow-up using a Seq2seq model which uses Bidirectional RNNs and Attention models.


### Installation
To run the chatbot program you must install the following libraries -
1. Install spacy 
```
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
4. Install os
```
pip install os
```
5. Install nltk
```
pip install nltk
```

### Training data
We have trained our model on the Cornell Movie Dialogue Corpus.
The files include -
movie_conversations.txt and movie_lines.txt


### Usage 
Note - First make sure that all the files required are saved in the same directory and all the required libraries are installed before running the program.
For Windows -
```
python basic_chatbot.py
```

For Linux (For Python3+ versions) -
```
python3 basic_chatbot.py
```

-----------------------------------------------------------------------------------------------------------------------------------

