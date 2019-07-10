import json

#Storing the json in "data"
with open("train.json", "r") as read_file:
    data = json.load(read_file)


#Execute below block to store on the names of the title in "write.txt"
'''
for i in range(442):
    f=open('topics.txt',"a+")
    f.write(data["data"][i]["title"]+"\n")
'''

#print(data)
#print(data["data"][0]['paragraphs'][0]["qas"][0]["question"])


from_file=open('topics.txt',"r")

#The final json is stored in "final.json"
final=open('final.json','a+')


list_of_topics=[]
for line in from_file.readlines():
    line=line.strip()
    list_of_topics.append(line)



dictionary={}
list_of_data=[]
t=data["data"]
k=0
count=0
for i in t:
    topic=i["title"]
    topic=topic.lower()
    dictionary[topic]=[]
    print(topic)
    j=i["paragraphs"]
    for k in j:
        k=k["qas"]
        for l in k:
            q=l['question']
            a=l['answers']
            if(len(a)==0):
                pass
            else:
                #print("question: ",q)
                q=q.lower()
                dictionary[topic].append(q)
                count+=1
                for z in a:
                    z=z["text"]
                 #   print("answer: ",z)
                    z=z.lower()
                    dictionary[topic].append(z)    


print(count)
print(len(dictionary))
json.dump(dictionary,final)

'''
for topic in list_of_topics:
    topics=list_of_data[k]
    paragraphs=topics["paragraphs"]
    qas=paragraphs[0]
    data_list=qas["qas"]
    for i in data_list:
        answers=i["answers"]
        if(len(answers)==0):
            pass
        else:
            print("queston: ",i["question"])
            answers=answers[0]["text"]
            print("answer: ",answers)
            count+=1
    k+=1
print(count)
#for line in from_file.readlines():
#    line=line.strip()
#    list_of_topics.append(line)
    # dict[line]=[]

for 



for line in from_file.readlines():
    k=0
    line=line.strip()
    dict[line]=[] #Creating the keys of the dictionary from the titles stored in write.txt

    for i in data["data"][k]['paragraphs'][0]["qas"]:  #Used to go through each element of first topic. Guess the error is here?

        dict[line].append(i["question"]) #One question
        dict[line].append(i['answers'][0]['text']) #And then its answer
        k+=1

json.dump(dict, final)
'''