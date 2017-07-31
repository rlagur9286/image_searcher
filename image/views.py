from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def my_sum(request, x, y):
    #request: HttpRequest
    return HttpResponse(int(x) + int(y))

def my_hello(request, name, age):
    #request: HttpRequest
    return HttpResponse('안녕하세요. {}. {}살 이시네요.'.format(name, age))