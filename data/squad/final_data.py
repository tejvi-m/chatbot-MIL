f1=open(file='questions.txt',mode='r',encoding='utf-8',errors='ignore')
f2=open(file='answers.txt',mode='r',encoding='utf-8',errors='ignore')
f3=open(file='final.txt',mode='a',encoding='utf-8',errors='ignore')
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
print('starting writing')
while(i<len(questions)):
    f3.write(questions[i])
    f3.write(answers[i])
    i+=1
print('Finished writing')

f1.close()
f2.close()
f3.close()
