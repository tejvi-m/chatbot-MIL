from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential #To integrate all kinds of layers one after another in a sequence
from keras.layers import Dense #One kind of layer
from keras.layers import LSTM #Another kind of layer which has memory cells
from keras.optimizers import RMSprop #optimizer
from keras.utils.data_utils import get_file
import pandas as pd
import numpy as np
import random
import sys
import io
from spacy.lang.en import English
from keras.models import load_model
from textblob import TextBlob


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

filename="mytext.txt"
text=open(filename).read()
#text=text.lower()

# Change: Removing unnecessary characters to learn by the neural network
text = [char for char in text if char not in ['\t','\n']]
text="".join(text)



#filename2="neg.txt"
#text1=open(filename2).read()
#text1=text1.lower()
text1="I don't like this bag"
#Sentiment analysis on that sentence
pol = sid.polarity_scores(text1)["compound"]
pol=pol*(-1)        #Final polarity
print("Final Polarity required", pol)

print('corpus length:', len(text))



chars = sorted(list(set(text)))
print('total chars:', len(chars))

#to encode
char_indices = dict((c, i) for i, c in enumerate(chars))
#to decode
indices_char = dict((i, c) for i, c in enumerate(chars))



# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))


#Vectorization
print('Vectorization...')


x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#print(y)



# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.04)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)




#def analyzer
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)




def on_epoch_end(epoch, _):
    file2=open('read.txt','a')
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)


        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            file2.write(next_char)
        print()
    file2.write('\n')
    file2.close()

    #Extract each sentence from the file into which we have written the generated characters and check if any of them has sentiment of required polarity
    file3=open('read.txt','r')
    nlp=English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc=nlp(file3.read())
    for sent in doc.sents:
        if(TextBlob((sent.string.strip())).sentiment.polarity==pol):
            print("polarity found")
            print(pol)
            print(sent)
            #sys.exit(0)
        else:
            print("Not found")
            #sys.exit(0)
    file3.close()
#print("Not found")
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=30,
callbacks=[print_callback])


#saving model
model.save('save.h5')
print("Model saved")


#loading model

#Sentiment analysis
