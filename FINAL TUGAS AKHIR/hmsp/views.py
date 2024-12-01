from django.shortcuts import render
from .proses_data import stock_list, predict_stock_price, prediction_history, prediction_detail

def callStock_list(request):
    return stock_list(request)

def home(request):
    return render(request, 'html/home.html')

def history(request):
    # return render(request, 'html/History.html')
    return prediction_history(request)

def historyDetail(request, predictionRef):
    # return render(request, 'html/HistoryDetail.html')
    return prediction_detail(request, predictionRef)

def predictStockPrice(request):
    # return render(request, 'html/predictionStockPrice.html')
    return predict_stock_price(request)

