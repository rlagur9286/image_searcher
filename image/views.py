from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def my_sum(request, x, y):
    #request: HttpRequest
    return HttpResponse(int(x) + int(y))
