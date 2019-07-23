from django.shortcuts import render

# Create your views here.
def options(request):
    return render(request, 'first_page.html')


def handler404(request):
    return render(request, '404.html', status=404)


def home(request):
    return render(request, 'home.html')
