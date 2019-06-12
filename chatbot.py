import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import re
import time
import collections
import os
import spacy
from spacy.matcher import Matcher
import sqlite3
import time



#Encoding
def build_dataset(words, n_words, atleast=1):
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words) #A counter is a container that stores elements as dictionary keys, and their counts are stored as dictionary values. Return a list of the n most common elements and their counts from the most common to the least. If n is omitted or None, most_common() returns all elements in the counter.
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words: #
        index = dictionary.get(word, 0)
        if index == 0:
            print(word)
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary



lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return ' '.join([i.strip() for i in filter(None, text.split())])



clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))



min_line_length = 2
max_line_length = 5
short_questions_temp = []
short_answers_temp = []

i = 0

for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

short_questions = []
short_answers = []

i = 0

for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1


#print(len(short_questions)) #22278
#print(len(short_answers)) #22278




question_test1=[]
i=0
while(i<100):
    user_response=input("User: ")
    question_test1.append(user_response)
    if(user_response=="Bye"):
        i=101
#question_test = short_questions[500:550]
answer_test1 = short_answers[500:550]
short_questions1 = short_questions[:500]
short_answers1 = short_answers[:500]







concat_from = ' '.join(short_questions1+question_test1).split() #splits all the questions into different words
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, vocabulary_size_from) #build_dataset function has been defined above
"""print('vocab from size: %d'%(vocabulary_size_from))
print('Most common words', count_from[4:10]) #Since 0=PAD, 1=GO , 2=EOS, 3=UNK and not the actual words
print('Sample data', data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print('filtered vocab size:',len(dictionary_from)) #661 #"filtered" since used set to create the dictionary
print("% of vocab used: {}%".format(round(len(dictionary_from)/vocabulary_size_from,4)*100))"""


concat_to = ' '.join(short_answers1+answer_test1).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
"""print('vocab from size: %d'%(vocabulary_size_to))
print('Most common words', count_to[4:10])
print('Sample data', data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print('filtered vocab size:',len(dictionary_to)) #664
print("% of vocab used: {}%".format(round(len(dictionary_to)/vocabulary_size_to,4)*100))"""



GO = dictionary_from['GO'] #So now it stores the value of the key "GO" in the dictionary
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']



for i in range(len(short_answers1)):
    short_answers1[i] += ' EOS'


class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size, learning_rate,
                 batch_size, dropout = 0.5, beam_width = 15):

        def lstm_cell(size, reuse=False):
            #returns input tensor or list of input tensors
            return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(),
                                           reuse=reuse)

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]

        # encoder
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = lstm_cell(size_layer // 2),
                cell_bw = lstm_cell(size_layer // 2),
                inputs = encoder_embedded,
                sequence_length = self.X_seq_len,
                dtype = tf.float32,
                scope = 'bidirectional_rnn_%d'%(n))
            encoder_embedded = tf.concat((out_fw, out_bw), 2)

        bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
        bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
        self.encoder_state = tuple([bi_lstm_state] * num_layers)

        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(num_layers))
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        # decoder
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        decoder_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)])
        dense_layer = tf.layers.Dense(to_dict_size)
        training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embeddings, decoder_input),
                sequence_length = self.Y_seq_len,
                time_major = False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cells,
                helper = training_helper,
                initial_state = self.encoder_state,
                output_layer = dense_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = decoder_embeddings,
                start_tokens = tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
                end_token = EOS)
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cells,
                helper = predicting_helper,
                initial_state = self.encoder_state,
                output_layer = dense_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
        self.training_logits = training_decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.sample_id
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


size_layer = 256
num_layers = 2
embedded_size = 128
learning_rate = 0.001
batch_size = 16
epoch = 5

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(size_layer, num_layers, embedded_size, len(dictionary_from),
                len(dictionary_to), learning_rate,batch_size)
sess.run(tf.global_variables_initializer())


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k,UNK))
        X.append(ints)
    return X


X = str_idx(short_questions1, dictionary_from)
Y = str_idx(short_answers1, dictionary_to)
X_test = str_idx(question_test1, dictionary_from)
Y_test = str_idx(answer_test1, dictionary_from)


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    for k in range(0, len(short_questions1), batch_size):
        index = min(k+batch_size, len(short_questions1))
        batch_x, seq_x = pad_sentence_batch(X[k: index], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD)
        predicted, accuracy,loss, _ = sess.run([model.predicting_ids,
                                                model.accuracy, model.cost, model.optimizer],
                                      feed_dict={model.X:batch_x,
                                                model.Y:batch_y})
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= (len(short_questions) / batch_size)
    total_accuracy /= (len(short_questions) / batch_size)
    print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))

'''
#model.save('mymodel.h5')


for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('REAL ANSWER:',' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')
'''

batch_x, seq_x = pad_sentence_batch(X_test[:batch_size], PAD)
batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD)
predicted = sess.run(model.predicting_ids, feed_dict={model.X:batch_x,model.X_seq_len:seq_x})

for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('REAL ANSWER:',' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')



#Follow up
#Use the question given
#Do ner-this is done
#Store it
#sleep(30)
#retreive it
#Done

#import spacy
#from spacy.matcher import Matcher

nlp=spacy.load('en_core_web_sm')
matcher=Matcher(nlp.vocab)      #initializing with shared vocab
question_test2=question_test1[0]
doc=nlp(question_test2)

#Writing a pattern that we're looking for
#proper noun,

pattern1=[{"POS": "PRON"}]
pattern2=[{"LEMMA": "lose"}]
pattern3=[{"POS":"NOUN"}]


#Adding pattern and application to doc
matcher.add("who",None, pattern1)
matcher.add("action",None, pattern2)
matcher.add("event",None, pattern3)

matches=matcher(doc)

#Printing text
#for match_id,start,end in matches:
 #   print("Match found:", doc[start:end].text)
list1=[]
for match_id, start, end in matches:
    rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
    span = doc[start : end]  # get the matched slice of the doc
    list1.append(span.text)
    print(rule_id, span.text)




#sql
#import sqlite3
print(list1)


who=list1[0]
action=list1[1]
event=list1[2]
#Creating and connecting to database
connection=sqlite3.connect("testdb1.db")

#Cursor
crsr=connection.cursor()

#SQLLLLLLLLL
#No primary keys
sql_command="""CREATE TABLE ner(who VARCHAR(20), action VARCHAR(20), event VARCHAR(20));"""
crsr.execute(sql_command)
crsr.execute("insert into ner (who, action, event) values (?, ?, ?)", (who, action, event))
connection.commit()
connection.close()



connection=sqlite3.connect("testdb1.db")
crsr=connection.cursor()
crsr.execute("select * from ner")

ans=crsr.fetchall()

for i in ans:
    print(i)






connection.close()

#time gap
time.sleep(30)

print('starting new')
os.system("python followup.py")




#DONE








































#PLEASE IGNORE THE FOLLOWING



'''
print("starting test.....")
time.sleep(10)
print("And now.....")

#Convert tuple to list
questionNew=list(ans[0])
questionNew=' '.join(questionNew)
#               #######START OF TEST#####


question_test=[]
question_test.append(questionNew)
print()
print()
print()
print(question_test)
print()
print()
print()

answer_test = short_answers[500:550]
short_questions = short_questions[:500]
short_answers = short_answers[:500]
    #In [4]:
concat_from = ' '.join(question_test+short_questions).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, vocabulary_size_from)
print('vocab from size: %d'%(vocabulary_size_from))
print('Most common words', count_from[4:10])
print('Sample data', data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print('filtered vocab size:',len(dictionary_from))
print("% of vocab used: {}%".format(round(len(dictionary_from)/vocabulary_size_from,4)*100))
#vocab from size: 657
#Most common words [('you', 132), ('is', 78), ('i', 68), ('what', 51), ('it', 50), ('that', 49)]
#Sample data [7, 28, 129, 35, 61, 42, 12, 22, 82, 225] ['what', 'good', 'stuff', 'she', 'okay', 'they', 'do', 'to', 'hey', 'sweet']
#filtered vocab size: 661
#% of vocab used: 100.61%
#In [5]:
concat_to = ' '.join(short_answers+answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print('vocab from size: %d'%(vocabulary_size_to))
print('Most common words', count_to[4:10])
print('Sample data', data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print('filtered vocab size:',len(dictionary_to))
print("% of vocab used: {}%".format(round(len(dictionary_to)/vocabulary_size_to,4)*100))
#vocab from size: 660
#Most common words [('i', 97), ('you', 91), ('is', 62), ('it', 58), ('not', 47), ('what', 39)]
#Sample data [12, 216, 5, 4, 94, 25, 59, 10, 8, 79] ['the', 'real', 'you', 'i', 'hope', 'so', 'they', 'do', 'not', 'hi']
#filtered vocab size: 664
#% of vocab used: 100.61%
#In [6]:
GO = dictionary_from['GO']
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']

for i in range(len(short_answers)):
    short_answers[i] += ' EOS'

#           ###END###



size_layer = 256
num_layers = 2
embedded_size = 128
learning_rate = 0.001
batch_size = 16
epoch = 5

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(size_layer, num_layers, embedded_size, len(dictionary_from),
                len(dictionary_to), learning_rate,batch_size)
sess.run(tf.global_variables_initializer())



X = str_idx(short_questions, dictionary_from)
Y = str_idx(short_answers, dictionary_to)
X_test = str_idx(question_test, dictionary_from)
Y_test = str_idx(answer_test, dictionary_from)
#In [13]:
maxlen_question = max([len(x) for x in X]) * 2
maxlen_answer = max([len(y) for y in Y]) * 2

#training again
for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    for k in range(0, len(short_questions), batch_size):
        index = min(k+batch_size, len(short_questions))
        batch_x, seq_x = pad_sentence_batch(X[k: index], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD)
        predicted, accuracy,loss, _ = sess.run([model.predicting_ids,
                                                model.accuracy, model.cost, model.optimizer],
                                      feed_dict={model.X:batch_x,
                                                model.Y:batch_y})
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= (len(short_questions) / batch_size)
    total_accuracy /= (len(short_questions) / batch_size)
    print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))






print("Done")
print()
print()
print()



for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')
print()
print()
print()

print("Done")
'''



'''
#Will try later
import spacy
import numpy as np
import pandas
import utils
nlp = spacy.load("en_core_web_sm")
def test(x):
    #due to the mapping between words and integers the DAE does not deal
    #well with numbers it has not seen before.
    #in the training phase, the prices have been split into '£' and the value
    #while doing NER.
    #this is just a quick fix, but there has to be a real fix for the test phase at least
    try:
        return vocab_to_int[x]
    except:
        try:
            return vocab_to_int['£'+x]
        except:
            print(x)
def newCorrupt(sentence):
    #get names - propernounse and sometimes nouns.
    #get adjectives
    #NER will get the cardinal values i.e., numbers.
    #get everything in NER
    #finally make the whole thing a set.
    #using NER doesn't add much to it right now
    sentence = nlp(sentence)
    toString = set()
    for ent in sentence.ents:
        try:
            x = float(ent.text)
            num = True
        except:
            num = False
        if num == False:
            for word in ent.text.split(' '):
                toString.add((word))
        else:
            toString.add((ent.text))
    for token in sentence:
        pos = token.pos_
        add = True
        if pos == 'PROPN':
            toString.add((token.text))
        if pos == "ADJ" or pos == "ADV":
            toString.add((token.text))
        if pos == "NOUN":
            add = np.random.choice([1, 0, 0, 1])
            if add:
                toString.add((token.text))
    curroptedS = ''
    for string in toString:
        curroptedS += ' ' + string
    return curroptedS
cor = trainset['ref'].apply(lambda x: newCorrupt(x))
trainset = trainset.assign(corrupted=cor)
as_tokens = trainset['corrupted'].apply(lambda x: [test(each) for each in x.split()])
trainset = trainset.assign(tokenized_corrupted=as_tokens)
trainset.to_csv('./data/processedTrainset.csv')
'''


'''
#Will try later
import spacy
import numpy as np
import pandas
import utils
nlp = spacy.load("en_core_web_sm")
def test(x):
    #due to the mapping between words and integers the DAE does not deal
    #well with numbers it has not seen before.
    #in the training phase, the prices have been split into '£' and the value
    #while doing NER.
    #this is just a quick fix, but there has to be a real fix for the test phase at least
    try:
        return vocab_to_int[x]
    except:
        try:
            return vocab_to_int['£'+x]
        except:
            print(x)
def newCorrupt(sentence):
    #get names - propernounse and sometimes nouns.
    #get adjectives
    #NER will get the cardinal values i.e., numbers.
    #get everything in NER
    #finally make the whole thing a set.
    #using NER doesn't add much to it right now
    sentence = nlp(sentence)
    toString = set()
    for ent in sentence.ents:
        try:
            x = float(ent.text)
            num = True
        except:
            num = False
        if num == False:
            for word in ent.text.split(' '):
                toString.add((word))
        else:
            toString.add((ent.text))
    for token in sentence:
        pos = token.pos_
        add = True
        if pos == 'PROPN':
            toString.add((token.text))
        if pos == "ADJ" or pos == "ADV":
            toString.add((token.text))
        if pos == "NOUN":
            add = np.random.choice([1, 0, 0, 1])
            if add:
                toString.add((token.text))
    curroptedS = ''
    for string in toString:
        curroptedS += ' ' + string
    return curroptedS
cor = trainset['ref'].apply(lambda x: newCorrupt(x))
trainset = trainset.assign(corrupted=cor)
as_tokens = trainset['corrupted'].apply(lambda x: [test(each) for each in x.split()])
trainset = trainset.assign(tokenized_corrupted=as_tokens)
trainset.to_csv('./data/processedTrainset.csv')
'''
