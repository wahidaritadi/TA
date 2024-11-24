from django.shortcuts import render
from .proses_data import stock_list

def callStock_list(request):
    return stock_list(request)

def home(request):
    return render(request, 'home/home.html')

