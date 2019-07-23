from django.shortcuts import render
import os
import datetime
from django.http import HttpResponse


print('hello')



def interaction(request):

    if(request.method=="POST"):

        user_text = request.POST['user_text']
        print(user_text)

        list_of_emo_conv=[]


        if(user_text!="Bye" and user_text!="bye"):
            reply, list_of_emo_conv = modelwork.callme(user_text)
        else:
            import time
            time.sleep(10)
            reply = modelwork.task3(list_of_emo_conv)


        print(reply)

        return HttpResponse(reply)

    else:
        doc=conversations.document(request.session['emailid'])

        return render(request, 'chatbot.html', {"trial":"Added"})

def handler500(request):
    return render(request, '500.html', status=500)

def handler404(request):
    return render(request, '404.html', status=404)
