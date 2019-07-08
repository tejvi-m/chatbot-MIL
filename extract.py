import json

#Storing the json in "data"
with open("train.json", "r") as read_file:
    data = json.load(read_file)


#Execute below block to store on the names of the title in "write.txt"
'''
for i in range(442):
    f=open('write.txt',"a+")
    f.write(data["data"][i]["title"]+"\n")


'''

#print(data)
#print(data["data"][0]['paragraphs'][0]["qas"][0]["question"])


from_file=open('write.txt',"r")

#The final json is stored in "final.json"
final=open('final.json','a+')

dict={}

for line in from_file.readlines():

    dict[line]=[] #Creating the keys of the dictionary from the titles stored in write.txt

    for i in data["data"][0]['paragraphs'][0]["qas"]:  #Used to go through each element of first topic. Guess the error is here?

        dict[line].append(i["question"]) #One question
        dict[line].append(i['answers'][0]['text']) #And then its answer


json.dump(dict, final)
