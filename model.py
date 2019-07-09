#importing all libraries
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from data.squad import data
from tensorlayer.models.seq2seq import Seq2seq
from seq2seq_attention import Seq2seqLuongAttention
import os
import sqlite3
import spacy
from textblob import TextBlob
import time
import matplotlib.pyplot as plt
import json
import spacy
from spacy.matcher import Matcher
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions, KeywordsOptions, SemanticRolesOptions, CategoriesOptions
natural_language_understanding=NaturalLanguageUnderstandingV1(version='2018-11-16',iam_apikey='KoTo6dvndPQEAy3T9LNqZMGJEHhEa2Yy3tHLyxTNO50r',url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')

nlp=spacy.load("en_core_web_md")

print('All libraries imported')

f=open("final.json", mode="r",encoding="utf-8",errors="ignore")
data1=json.load(f)
f.close()




#####MATCHER PATTERN#####
matcher=Matcher(nlp.vocab)
pattern=[{'POS':'NOUN'}]
matcher.add('PNOUN_PATTERN',None,pattern)


#
# print(type(f.readlines()))
'''
list_of_questions=[]
l=q.readlines()
for line in l:
    line=line.strip()
    list_of_questions.append(line)

list_of_answers=[]
l=a.readlines()
for line in l:
    line=line.strip()
    list_of_answers.append(line)
q.close()
a.close()
'''
def initial_setup(data_corpus):
    metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus)) 
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY

def classification(user_input):
    response=natural_language_understanding.analyze(text=user_input,
    features=Features(categories=CategoriesOptions(limit=1))).get_result()
    #print(json.dumps(response,indent=2))
    categories=response["categories"]
    #taking the highest score
    category=categories[0]
    label=category["label"]
    label=label.split("/")
    topic=label[1]
    print("topic: ",topic)
    return topic

def sentiment_extraction(user_input):
    sentiment=natural_language_understanding.analyze(text=user_input,
    features=Features(sentiment=SentimentOptions(text))).get_result()
    dic=sentiment["sentiment"]
    doc=dic["document"]
    score=doc["score"]
    print("sentiment: ",score)
    return score


def keyword_extraction(user_input):
    splitted=user_input.split()
    subject=''
    if(len(splitted)>3):
        keywords=natural_language_understanding.analyze(text=user_input,
        features=Features(semantic_roles=SemanticRolesOptions())).get_result()
        print(json.dumps(keywords,indent=2))
        
        l=keywords["semantic_roles"]
        if(len(l)!=0):
            if "i" in splitted:
                semantic_roles=keywords["semantic_roles"]
                ob=semantic_roles[0]
                subject=ob["object"]
                subject=subject["text"]
                print("subject: ",subject)
            else:
                semantic_roles=keywords["semantic_roles"]
                sub=semantic_roles[0]
                subject=sub["subject"]
                subject=subject["text"]
                print("subject: ",subject)
        else:
            pass
    else:
        doc=nlp(user_input)
        matches=matcher(doc)
        for match_id, start,end in matches:
            print("subject: ",doc[start:end].text)
            subject=doc[start:end].text

    list_of_sub=subject.split()
    return list_of_sub

def get_topics(dictionary):
    return dictionary.keys()

def intersection(lst1,lst2):
    return set(lst1).intersection(lst2)


def check(user_input, list_of_topics_initial):
    print("Searching in initial topics")
    text1=nlp(user_input)
    sentence=''
    list_of_sub=keyword_extraction(user_input)
    common=intersection(list_of_topics_initial,list_of_sub)
    common=list(common)
    common=''.join(common)
    print("common: ",common)
    if (common==''):
        pass
    else:
        for topic in list_of_topics_initial:
            if(common==topic or common in topic):
                qa=data1[topic]
                for i in range(0,len(qa),2):
                    text2=qa[i]
                    text2=nlp(text2)
                    if(text2.similarity(text1)>=0.9):
                        sentence=qa[i+1]
                    else:
                        pass
        print("sentence: ",sentence)
    return sentence

def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])



if __name__ == "__main__":
    data_corpus = "squad"

    #data preprocessing
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    batch_size = 32
    n_step = src_len // batch_size
    src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
    emb_dim = 1024

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word

    unk_id = word2idx['unk']   # 1
    pad_id = word2idx['_']     # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2

    #num_epochs = 50
    vocabulary_size = src_vocab_size
    
    count=0 #For keeping count of entries into db

    def inference(seed, top_n):
        model_.eval()
        seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
        sentence_id = model_(inputs=[[seed_id]], seq_length=20, start_token=start_id, top_n = top_n)
        sentence = []
        for w_id in sentence_id[0]:
            w = idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence

    decoder_seq_length = 20
    blacklist = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
    whitelist = '0123456789abcdefghijklmnopqrstuvwxyz '
    '''
    model_=model_ = Seq2seqLuongAttention(
            hidden_size=128, cell=tf.keras.layers.SimpleRNNCell,
            embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size,
                                                embedding_size=emb_dim), method='dot'
)
    '''

    
    model_ = Seq2seq(
        decoder_seq_length = decoder_seq_length,
       cell_enc=tf.keras.layers.LSTMCell,
      cell_dec=tf.keras.layers.LSTMCell,
     n_layer=3,
     n_units=256,
     embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim))
    




    
    # Uncomment below statements if you have already saved the model

    # load_weights = tl.files.load_npz(name='model.npz')
    tl.files.load_hdf5_to_weights('model.hdf5', model_, skip=False)
    #print('loaded model')
    #print('loaded model')
    #tl.files.assign_weights(load_weights, model_)

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    #+
    model_.train()


    list_of_topics_initial=get_topics(data1)
    #print(list_of_topics_initial)
    list_of_topics_initial=list(list_of_topics_initial)
    #print(list_of_topics_initial)
    #print(type(list_of_topics_initial))
    
#    db=sqlite3.connect('chatbot.db')
 #   cursor=db.cursor()
 #   cursor.execute('''create table user_inputs(questions TEXT)''')
#    db.commit()
    
    #seeds = ["happy birthday have a nice day",
     #           "donald trump won last nights presidential debate according to snap online polls"]

    #for seed in seeds:
      #      print("Query >", seed)
       #     top_n = 3
        #    for i in range(top_n):
         #       sentence = inference(seed, top_n)
          #      print(" >", ' '.join(sentence))
    '''
    for epoch in range(num_epochs):
        model_.train()
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        total_loss, n_iter = 0, 0
        for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                        total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=decoder_seq_length)
            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=decoder_seq_length)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            with tf.GradientTape() as tape:
                ## compute outputs
                output = model_(inputs = [X, _decode_seqs])
                
                output = tf.reshape(output, [-1, vocabulary_size])
                ## compute loss and update model
                loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

                grad = tape.gradient(loss, model_.all_weights)
                optimizer.apply_gradients(zip(grad, model_.all_weights))
            
            total_loss += loss
            n_iter += 1

        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))
        tl.files.save_weights_to_hdf5('model.hdf5', model_)
        print("model saved")   
    '''


    ##################################################################################################
    #Good till here
    dictionary={}
    list_of_topics_fp=[]
    ans=[]

    


    while(1):
        user_input=input("Enter query: ")
        user_input=filter_line(user_input,whitelist)
        num_word=len(user_input.split())
        print(user_input)
        print(num_word)
        sent=check(user_input,list_of_topics_initial)

        if(user_input=="Bye" or user_input=="bye"):
            print("Bye")
            break
        elif(num_word<4):
            if(sent!=''):
                print("Searching")
                #sentence=''.join(sent)
                print(">",sent)
                sent=''
            else: 
                print("Generating")   
                sentence=inference(user_input,1)
                print(">",' '.join(sentence))
        else:
            if(sent!=''):
                print("Searching")
                topic=classification(user_input)
                if(topic not in list_of_topics_fp):
                    list_of_topics_fp.append(topic)
                    dictionary[topic]=[user_input]
                else:
                    dictionary[topic].append(user_input)
                #sentence=''.join(sent)
                print(">",sent)
                sent=''
            else:
                print("Generating")
                topic=classification(user_input)
                if(topic not in list_of_topics_fp):
                    list_of_topics_fp.append(topic)
                    dictionary[topic]=[user_input]
                else:
                    dictionary[topic].append(user_input)
                            
                print("Query >", user_input)
                top_n=1
                for i in range(top_n):
                    sentence=inference(user_input, top_n)
                    print(">",' '.join(sentence))

    
#Follow up
#Follow up
    print('Beginning of follow up.....')
    while(1):
        user_input=input("Enter text: ")
        text1=nlp(user_input)
        topic=classification(user_input)
        if(topic not in list_of_topics_fp):
            list_of_topics_fp.append(topic)
            dictionary[topic]=[user_input]
        else:
            list_of_conv=dictionary[topic]
            for text in list_of_conv:
                num_word2=len(user_input.split())
                if num_word2<4:
                    pass
                else:
                    sentiment=sentiment_extraction(text)
                    text2=nlp(text)
                    if(text2.similarity(text1)>=0.8 and sentiment<0):
                        print(text,sentiment,user_input)
                        #Do follow up
                        print("Bot> ", user_input)
                        sentence=inference(user_input,1)
                        print(">",' '.join(sentence))
                    else:
                        #Generate new question
                        pass



#Follow up
#create db, input respones to db
#if db is negative, store it send it back to model afte some time decided by the decision system
'''
    
    print('FOLLOW UP BEGINNING.......')
        #insert into table
    #db.close()

    #New start
    #follow up
    #Loading model
    load_weights = tl.files.load_hdf5_to_weights('model.hdf5', model_, skip=False)
    #tl.files.assign_weights(load_weights, model_)
    
    #Really need to do the decision system
    time.sleep(5)

    print('Beginning of follow up')
    #Connect to db and take in inputs
    db=sqlite3.connect('chatbot.db')
    cursor=db.cursor()
    inputs=cursor.fetchall()
    
    seeds=[]
    for i in range(count):
        a=list(inputs[i])
        a=''.join(a)
        seeds.append(a)

    for seed in seeds:
            print("Query >", seed)
            top_n = 3
            for i in range(top_n):
                sentence = inference(seed, top_n)
                print(" >", ' '.join(sentence))
'''


