#Start of generalization
#Convert into txt files
#Basically, extract from csv into txt
import pandas as pd
import json
import os

choice=input('Enter your choice: ')
name=input('Enter name of file without extension: ')
if (choice=='csv'):
    fname=name+".csv"
    print('Extracting values from csv and converting to text......')
    print("Don't panic, takes some time")
    f=open(file='data.txt',mode='a',encoding='utf-8',errors='ignore')
    df=pd.read_csv(fname,usecols=["text"])
    print(type(df))
    a=df.values.tolist()
    print(type(a))
    final=a[:10000]
    last=[]
    for i in final:
        string="".join(i)
        last.append(string)
        string=''


    print(len(last))
    print(type(last))
    for i in last:
        f.write(i)
        f.write('\n')

    f.close()
    
elif(choice=='json'):
    fname=name+'.json'
    print('Extracting values from json and converting to text......')
    #f=open(file='final.txt',mode='a',encoding='utf-8',errors='ignore')
    answers=open(file='answers.txt',mode='a',encoding='utf-8',errors='ignore')    
    questions=open(file='questions.txt',mode='a',encoding='utf-8',errors='ignore')
    input=open(fname,encoding='utf-8',errors='ignore')
    json_dec=json.load(input)
    #squad=open('squad.txt',encoding='utf-8',errors='ignore')
    print('Opening successful')    
    
    t=json_dec['data']
    #print(type(t[0]))  
    count=0
    #For questions
    print('Writing into questions')
    for i in t:
        j=i['paragraphs']
        for k in j:
            k=k['qas']
            for l in k:
                q=l['question']
                z=l['answers']
                if (len(z)==0):
                    pass
                else:
                    questions.write(q)
                    questions.write('\n')
                    count+=1
    print()
    print(count)

    count=0
    for i in t:
        j=i['paragraphs']
        for k in j:
            k=k['qas']
            for l in k:
                l=l['answers']
                if (len(l)==0):
                    pass
                else:
                    for z in l:
                        count+=1
                        z=z['text']
                        answers.write(z)
                        answers.write('\n')
                            
    answers.close()
    questions.close()
    print('Converting to required format......')
    f1=open(file='questions.txt',mode='r',encoding='utf-8',errors='ignore')
    f2=open(file='answers.txt',mode='r',encoding='utf-8',errors='ignore')
    f3=open(file='data.txt',mode='a',encoding='utf-8',errors='ignore')
    questions=[]
    answers=[]
    #a=f1.readline()
    #print(a.strip())

    #print(f1.readline())

    for question in f1:
        questions.append(question)

    for answer in f2:
        answers.append(answer)

    print(len(questions))
    print(len(answers))
    i=0
    print('starting writing........')
    while(i<len(questions)):
        f3.write(questions[i])
        f3.write(answers[i])
        i+=1
    print('Finished writing')

    f1.close()
    f2.close()
    f3.close()
    os.remove('questions.txt')
    print('Removed questions text file')
    os.remove('answers.txt')
    print('Removed answers text file')
    print('Final text is stored in data.txt....')

else:
    pass


print('Execution complete')

print('Starting process to convert into metadata and npy....')
os.system("data.py 1")

print('Finished generating metadata and npy')
os.remove('data.txt')
print('Removed data.txt')



