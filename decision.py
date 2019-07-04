#Decision system
'''
Store it into topics
Maybe like 10 conversations different topics
Once you get topics, start next
step
Follow up: check current topic, check similarity
If similarity greater than 0.8
Follow up
Otherwise 
Create question
'''
import json
import spacy
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions, KeywordsOptions, SemanticRolesOptions, CategoriesOptions
natural_language_understanding=NaturalLanguageUnderstandingV1(version='2018-11-16',iam_apikey='KoTo6dvndPQEAy3T9LNqZMGJEHhEa2Yy3tHLyxTNO50r',url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')

nlp=spacy.load("en_core_web_md")

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

#user_input="i am feeling sick"
#classification(user_input)

dictionary={}
list_of_topics=[]

while(1):
    user_input=input("Enter text: ")
    if(user_input=='Bye' or user_input=="bye"):
        break
    else:
        #Starting to classify
        topic=classification(user_input)
        if(topic not in list_of_topics):
            list_of_topics.append(topic)
            dictionary[topic]=[user_input]
        else:
            dictionary[topic].append(user_input) 

print(dictionary)

#Follow up
print('Beginning of follow up.....')
while(1):
    user_input=input("Enter text: ")
    text1=nlp(user_input)
    topic=classification(user_input)
    list_of_conv=dictionary[topic]
    for text in list_of_conv:
        sentiment=sentiment_extraction(text)
        text2=nlp(text)
        if(text2.similarity(text1)>=0.8 and sentiment<0):
            print(text,sentiment,user_input)
            #Do follow up
            pass
        else:
            #Generate new question
            pass
