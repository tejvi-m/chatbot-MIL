from django.shortcuts import render

# Create your views here.
def personalize(request):
    if(request.method=="POST"):
        question = request.POST.get('question')
        answer = request.POST.get('answer')

    
        print(question, answer)
        return render(request, 'personalize.html',{"success":question})
    else:
        return render(request , 'personalize.html')
