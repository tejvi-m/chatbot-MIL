import conversation.python_files.trymodel as trymodel
from conversation.python_files.trymodel import data1


##################################################################################################

list_of_topics_initial=trymodel.get_topics(data1)
list_of_topics_initial=list(list_of_topics_initial)

#Dictionaries and lists used to store and classify conversations
dictionary={}
list_of_topics_fp=[]
ans=[]
list_of_emo_convo=[]
fallback_intent=['I didn\'t get that. Can you say it again?','I missed what you said. What was that?','Sorry, could you say that again?','Sorry, can you say that again?','Can you say that again?','Sorry, I didn\'t get that. Can you rephrase?','Sorry, what was that?','One more time?','What was that?','Say that one more time?','I didn\'t get that. Can you repeat?','I missed that, say that again?']
ending_convo=['bye', 'see you','goodbye','Bye', 'hasta la vista','i\'ll be back']
game_request=["yea","okay", "yep", "yeah", "yes","sure"]
no_response=["no","nope","nahh"]
blacklist = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
whitelist = '0123456789abcdefghijklmnopqrstuvwxyz '

def callme(user_input):
    sen=trymodel.sentiment_extraction(user_input)
    if(sen<0):
        #print("sentiment: ",sen)
        list_of_emo_convo.append(user_input)
        print("emo",list_of_emo_convo)
    #print("movie convo response: ",' '.join(inference(user_input,1)))
    df_response=trymodel.detect_intent_texts('lustrous-jet-246503','unique',user_input,'en')

    if "joke" in df_response:
        print(">",df_response)
        #txt2spch(user_input)
        user_input=input("Enter query: ")
        user_input=filter_line(user_input, whitelist)
        if user_input in no_response:
            df_response=detect_intent_texts('lustrous-jet-246503','unique',user_input,'en')
        else:
            user_input=user_input+ " -jk"
            df_response=detect_intent_texts('lustrous-jet-246503','unique',user_input,'en')
        return df_response, list_of_emo_convo

    else:
        if(user_input in ending_convo):
            if(df_response in fallback_intent):
                return('bye',list_of_emo_convo)
            else:
                return (df_response)
        else:
            pass

    if df_response in fallback_intent:
        user_input=trymodel.filter_line(user_input,whitelist)
        list_of_keywords=trymodel.keyword_extraction(user_input)
        keywords=' '.join(list_of_keywords)
        list_of_user_input=user_input.split()
        if("what" in list_of_user_input and "is" in list_of_user_input and len(list_of_user_input)<=4):
            keywords=list_of_user_input[-2:]
            keywords=' '.join(list_of_user_input)
            wiki=trymodel.wiki_extract(keywords)
            dictionary_wiki=trymodel.info_extraction(wiki,user_input)
            wiki_keywords=trymodel.idk(user_input,dictionary_wiki, wiki)
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
            wiki=trymodel.wiki_extract(keywords)
            dictionary_wiki=trymodel.info_extraction(wiki,user_input)
            wiki_keywords=trymodel.idk(user_input,dictionary_wiki, wiki)
            if(type(wiki_keywords)==list):
                doc=trymodel.nlp_sent(wiki)
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


        sentiment=trymodel.sentiment_extraction(user_input)
        num_word=len(user_input.split())
        if(wiki==''):
            if(num_word>3 and sentiment<0):
                #list_of_emo_convo.append(user_input)
                pass
            else:
                pass
            sent=trymodel.check(user_input,list_of_topics_initial)

            if(user_input=="Bye" or user_input=="bye"):
                return "Bye",list_of_emo_convo
            elif(num_word<4):
                if(sent!=''):
                    print("Searching")
                    return sent,list_of_emo_convo
                    sent=''
                else:
                    print("Generating")
                    sentence=trymodel.inference(user_input,1)
                    return ' '.join(sentence),list_of_emo_convo
            else:
                if(sent!=''):
                    print("Searching")
                    topic=trymodel.classification(user_input)
                    if(topic not in list_of_topics_fp):
                        list_of_topics_fp.append(topic)
                        dictionary[topic]=[user_input]
                    else:
                        dictionary[topic].append(user_input)
                    return sent, list_of_emo_convo
                    sent=''
                else:
                    print("Generating")
                    topic=trymodel.classification(user_input)
                    if(topic not in list_of_topics_fp):
                        list_of_topics_fp.append(topic)
                        dictionary[topic]=[user_input]
                    else:
                        dictionary[topic].append(user_input)

                    print("Query >", user_input)
                    top_n=1
                    for i in range(top_n):
                        sentence=trymodel.inference(user_input, top_n)
                        return ' '.join(sentence), list_of_emo_convo
        else:
            return wiki, list_of_emo_convo

    else:
        return df_response, list_of_emo_convo





def task3(list_of_emo_conv):
    import time
    print("Threading for responses")
    #resp()
    time_taken, sed_txt=trymodel.time_delay(list_of_emo_convo)
    print("Sleeping......")
    print(sed_txt)
    if sed_txt=='':
        print("No follow up required")
        #resp()
        print("Thank you")
        pass
    else:
        print('Beginning of follow up.....')
        df_response_fp=trymodel.detect_intent_texts('lustrous-jet-246503','unique',sed_txt,'en')
        if "What happened" in df_response_fp:
            sed_txt=sed_txt +" -fp"
            df_response_fp=trymodel.detect_intent_texts('lustrous-jet-246503','unique',sed_txt,'en')
            time.sleep(time_taken)
            return df_response_fp
            callme('hello')
            print("Thank you!")
        else:
            print("No follow up required")
            callme('hello')
            print("Thank you")
