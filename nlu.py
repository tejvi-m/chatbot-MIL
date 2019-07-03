import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions, KeywordsOptions, SemanticRolesOptions
natural_language_understanding=NaturalLanguageUnderstandingV1(version='2018-11-16',iam_apikey='KoTo6dvndPQEAy3T9LNqZMGJEHhEa2Yy3tHLyxTNO50r',url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')

text=input('Enter text: ')
sentiment=natural_language_understanding.analyze(text=text,
    features=Features(sentiment=SentimentOptions(text))).get_result()
keyword=natural_language_understanding.analyze(text=text,features=Features(keywords=KeywordsOptions())).get_result()
semantics=response = natural_language_understanding.analyze(
    text=text,
    features=Features(semantic_roles=SemanticRolesOptions())).get_result()
#print(json.dumps(semantics,indent=2))


#Extracting score
def sentiment_extraction(sentiment):
    dic=sentiment["sentiment"]
    doc=dic["document"]
    score=doc["score"]
    print("sentiment: ",score)
    return score

#Extracting object and relevance to conversation
def keyword_extraction(keyword):
    lis=keyword["keywords"]
    dic2=lis[0]
    text=dic2["text"]
    relevance=dic2["relevance"]
    print("keyword: ", text, "relevance: ", relevance)
    return text,relevance

def subverb_extraction(semantics):
    #Extracting subject and verb
    lis2=semantics["semantic_roles"]
    dic3=lis2[0]
    subject=dic3["subject"]
    sub_text=subject["text"]
    action=dic3["action"]
    verb=action["verb"]
    verb_text=verb["text"]
    print("subject: ",sub_text)
    print("verb: ", verb_text)
    return sub_text,verb_text


sentiment_extraction(sentiment)
keyword_extraction(keyword)
subverb_extraction(semantics)