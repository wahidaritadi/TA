from django.shortcuts import render
from .proses_data import stock_list

def callStock_list(request):
    return stock_list(request)

def home(request):
    return render(request, 'html/home.html')

def predictData(request):
    return render(request, 'html/predictionStockPrice.html')

