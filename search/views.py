from django.shortcuts import render
# Create your views here.


def post_list(request):
    return render(request, 'search/post_list.html')

def my_hello(request, name):
    return render(request, 'search/post_list.html', {'name': name})

