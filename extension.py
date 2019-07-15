import pandas as pd
import csv
raw_data={'text':[]}

#f=open('extension.csv',mode='a+',encoding='utf-8',errors='ignore')
print("Type 'DONE' to exit")
while(1):
    user_input_ques=input("Enter question: ")
    if(user_input_ques=='Done' or user_input_ques=='done' or user_input_ques=='DONE'):
        print('Thank you!')
        break
    else:
        user_input_ans=input("Enter answer: ")
        raw_data['text'].append(user_input_ques)
        raw_data['text'].append(user_input_ans)
        

df=pd.DataFrame(raw_data,columns=['text'])
df.to_csv("extension.csv",index=False)










    