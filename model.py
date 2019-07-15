#importing all libraries
#from functions import detect_intent_texts, initial_setup, classification, sentiment_extraction, keyword_extraction, get_topics, intersection, check, filter_line, time_delay,wiki_extract, info_extraction,idk
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
import random
import spacy
from textblob import TextBlob
import time
import wikipedia
import json
import spacy
import dialogflow_v2 as dialogflow
from spacy.lang.en import English
from spacy.matcher import Matcher
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions, KeywordsOptions, SemanticRolesOptions, CategoriesOptions
from concurrent.futures import ThreadPoolExecutor
import threading



def detect_intent_texts(project_id, session_id, text, language_code):
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)

        if text:
            text_input = dialogflow.types.TextInput(
                text=text, language_code=language_code)
            query_input = dialogflow.types.QueryInput(text=text_input)
            response = session_client.detect_intent(
                session=session, query_input=query_input)

            return response.query_result.fulfillment_text

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
    splitted=user_input.split()
    if(len(splitted)>3):
        response=natural_language_understanding.analyze(text=user_input,
        features=Features(categories=CategoriesOptions(limit=1))).get_result()
        categories=response["categories"]
        try:
            category=categories[0]
            label=category["label"]
            label=label.split("/")
            topic=label[1]
            return topic
        except:
            return "None"
    else:
        return "None"


def sentiment_extraction(user_input):
    splitted=user_input.split()
    if(len(splitted)<4):
        blob=TextBlob(user_input)
        score=blob.sentiment.polarity
        return score
    else:
        sentiment=natural_language_understanding.analyze(text=user_input,
        features=Features(sentiment=SentimentOptions(user_input))).get_result()
        dic=sentiment["sentiment"]
        doc=dic["document"]
        score=doc["score"]
        return score


def keyword_extraction(user_input):
    user_input=user_input.strip()
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
            else:
                semantic_roles=keywords["semantic_roles"]
                sub=semantic_roles[0]
                subject=sub["subject"]
                subject=subject["text"]
        else:
            matcher=Matcher(nlp.vocab)
            pattern=[{'POS':'NOUN'}]
            matcher.add('NOUN_PATTERN',None,pattern)
            doc=nlp(user_input)
            for token in doc:
                print(token.text,token.pos_)
            matches=matcher(doc)
            subs=[]
            for match_id, start,end in matches:
                print("subject: ",doc[start:end].text)
                subs.append(doc[start:end].text)
            subject=' '.join(subs)
            
    else:
        matcher=Matcher(nlp.vocab)
        pattern=[{'POS':'NOUN'}]
        matcher.add('NOUN_PATTERN',None,pattern)
        doc=nlp(user_input)
        for token in doc:
                print(token.text,token.pos_)
        matches=matcher(doc)
        subs=[]
        for match_id, start,end in matches:
            subs.append(doc[start:end].text)
        subject=' '.join(subs)
            

    list_of_sub=subject.split()
    print(list_of_sub)
    return list_of_sub



def get_topics(dictionary):
    return dictionary.keys()



def intersection(lst1,lst2):
    return set(lst1).intersection(lst2)



def check(user_input, list_of_topics_initial):
    text1=nlp(user_input)
    sentence=''
    list_of_sub=keyword_extraction(user_input)
    common=intersection(list_of_topics_initial,list_of_sub)
    common=list(common)
    common=''.join(common)
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



def time_delay(list_of_emo_convo):
    list_of_scores=[]
    for i in list_of_emo_convo:
        score=sentiment_extraction(i)
        list_of_scores.append(score)
    neg=min(list_of_scores)
    neg_index=list_of_scores.index(neg)
    sad_text=list_of_emo_convo[neg_index]
    time_delay=neg*10*-1
    print("time delay:",time_delay,"s")
    return time_delay, sad_text
    '''
    if(len(list_of_emo_convo)<=10):
        for i in list_of_emo_convo:
            sentiment=sentiment_extraction(i)
            if(sentiment>sed_one):
                sed_one=sentiment
                sed_statement=i
    else:
        sed_statement=list_of_emo_convo[-1]
        sed_one=sentiment_extraction(sed_statement)
    time_delay=sed_one*100
    return time_delay
    '''

def wiki_extract(keywords):
    try:
        wiki=wikipedia.summary(keywords)
        wiki_list=wiki.split('.')

        return wiki
    except:
        wiki=''
        return wiki

def idk(user_input,dictionary,wiki):
    imp_stuff=[]
    wiki_list=wiki.split('.')
    if("where" in user_input):
        for i in dictionary:
            if(dictionary[i]=="GPE"):
                print(i)
                imp_stuff.append(i)
        
        return imp_stuff
    elif("what" in user_input):
        return '.'.join(wiki_list[:3])
    elif("when" in user_input):
        for i in dictionary:
            if(dictionary[i]=="DATE"):
                print(i)
                imp_stuff.append(i)
        return imp_stuff     
    elif("who" in user_input):
        return wiki          #Need to change it
    elif("why" in user_input):
        return '.'.join(wiki_list[:3])
    elif("which" in user_input):
        #idk
        return '.'.join(wiki_list[:3])
    elif("is"==user_input[0]):
        if(random.random()>0.5):
            answer='yes'
            return answer
        else:
            answer='no'
            return answer
    else:
        wiki=''
        return wiki
    


def info_extraction(wiki,user_input):
    list_of_wiki_inputs=wiki.split('.')
    dictionary={}
    for text in list_of_wiki_inputs:
        doc=nlp(text)
        for ent in doc.ents:
            dictionary[ent.text]=ent.label_
    return dictionary



#MULTI THREADING
def task():
    print("Executing task on different threads")
    natural_language_understanding=NaturalLanguageUnderstandingV1(version='2018-11-16',iam_apikey='#############################',url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')
    nlp=spacy.load("en_core_web_md")
    nlp_sent=English()
    sentencizer=nlp_sent.create_pipe("sentencizer")
    nlp_sent.add_pipe(sentencizer)
    return natural_language_understanding, nlp, nlp_sent,sentencizer,nlp_sent

executor=ThreadPoolExecutor(max_workers=8)
natural_language_understanding, nlp, nlp_sent,sentencizer,nlp_sent=executor.submit(task).result()


print('All libraries imported')

#Opening final json
f=open("final.json", mode="r",encoding="utf-8",errors="ignore")
data1=json.load(f)
f.close()


#Google dialogflow
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='#################'


        

    


if __name__ == "__main__":
    data_corpus = "cornell_corpus"


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

    def task2():
        print("Threading for loading model")
    # Uncomment the below statement if you have already saved the model or comment it if you want to train model
        model_ = Seq2seq(decoder_seq_length = decoder_seq_length,
                    cell_enc=tf.keras.layers.LSTMCell,
                    cell_dec=tf.keras.layers.LSTMCell,
                    n_layer=3,
                    n_units=256,
                    embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim))
    
    
        tl.files.load_hdf5_to_weights('models/model_conv.hdf5', model_, skip=False)
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        model_.train()
        return model_
    executor=ThreadPoolExecutor(max_workers=8)
    model_=executor.submit(task2).result()
    

    list_of_topics_initial=get_topics(data1)
    list_of_topics_initial=list(list_of_topics_initial)
    
    
    #Uncomment for training

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

        #   printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))
        tl.files.save_weights_to_hdf5('model.hdf5', model_)
        print("model saved")   
    
    '''
    ##################################################################################################
    #Dictionaries and lists used to store and classify conversations
    dictionary={}
    list_of_topics_fp=[]
    ans=[]
    list_of_emo_convo=[]
    fallback_intent=['I didn\'t get that. Can you say it again?','I missed what you said. What was that?','Sorry, could you say that again?','Sorry, can you say that again?','Can you say that again?','Sorry, I didn\'t get that. Can you rephrase?','Sorry, what was that?','One more time?','What was that?','Say that one more time?','I didn\'t get that. Can you repeat?','I missed that, say that again?']
    ending_convo=['bye', 'see you','goodbye','Bye', 'hasta la vista','i\'ll be back']


    
    def resp():
        while(1):
            user_input=input("Enter query: ")
            sen=sentiment_extraction(user_input)
            if(sen<0):
                print("sentiment: ",sen)
                list_of_emo_convo.append(user_input)
            #print("movie convo response: ",' '.join(inference(user_input,1)))
            df_response=detect_intent_texts('chatbot-oydwtm','unique',user_input,'en')
            if(user_input in ending_convo):
                if(df_response in fallback_intent):
                    print("Bye")
                    break
                else:
                    print(">",df_response)
                    break
            else:
                pass
            
            if df_response in fallback_intent:
                user_input=filter_line(user_input,whitelist)
                list_of_keywords=keyword_extraction(user_input)
                keywords=' '.join(list_of_keywords)
                list_of_user_input=user_input.split()
                if("what" in list_of_user_input and "is" in list_of_user_input and len(list_of_user_input)<=4):
                    keywords=list_of_user_input[-2:]
                    keywords=' '.join(list_of_user_input)
                    wiki=wiki_extract(keywords)
                    dictionary_wiki=info_extraction(wiki,user_input)
                    wiki_keywords=idk(user_input,dictionary_wiki, wiki)
                    if(type(wiki_keywords)==list):
                        for text in wiki.split('.'):
                            count=0
                            for key in wiki_keywords:
                                if key in text:
                                    count+=1
                            if count>=1:
                                list1=[]
                                list1.append(text)
                                wiki=''.join(list1)           
                                break
                    else:
                        wiki=wiki_keywords
                            


                else:
                    wiki=wiki_extract(keywords)
                    dictionary_wiki=info_extraction(wiki,user_input)
                    wiki_keywords=idk(user_input,dictionary_wiki, wiki)
                    if(type(wiki_keywords)==list):
                        doc=nlp_sent(wiki)
                        if "who" in user_input:
                            for sent in doc.sents:
                                count=0
                                for key in list_of_keywords:
                                    if key in text:
                                        count+=1
                                if(count>=1):
                                    list1=[]
                                    list1.append(sent)
                                    wiki='.'.join(list1)
                        else:
                            for text in wiki.split('.'):
                                count=0
                                for key in wiki_keywords:
                                    if key in text:
                                        count+=1
                                if count>=1:
                                    list1=[]
                                    list1.append(text)
                                    wiki=''.join(list1)           
                                    break
                    else:
                        wiki=wiki_keywords
                

                sentiment=sentiment_extraction(user_input)
                num_word=len(user_input.split())
                if(wiki==''):
                    if(num_word>3 and sentiment<0):
                        #list_of_emo_convo.append(user_input)
                        pass
                    else:
                        pass
                    sent=check(user_input,list_of_topics_initial)

                    if(user_input=="Bye" or user_input=="bye"):
                        print("Bye")
                        break
                    elif(num_word<4):
                        if(sent!=''):
                            print("Searching")
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
                else:
                    print("second last else>",wiki)
            
            else:
                print(">",df_response)
        return 0
    def task3():
        print("Threading for resonses")
        resp()
        time_taken, sed_txt=time_delay(list_of_emo_convo)
        print("Sleeping......")
        print(sed_txt)
        time.sleep(time_taken)
    
        print('Beginning of follow up.....')
        sed_txt=sed_txt +" -fp"
        df_response_fp=detect_intent_texts('chatbot-oydwtm','unique',sed_txt,'en')
        print("follow up: ",df_response_fp)
        resp()  
        print("Thank you!")  
    executor=ThreadPoolExecutor(max_workers=8)
    
    executor.submit(task3).result()