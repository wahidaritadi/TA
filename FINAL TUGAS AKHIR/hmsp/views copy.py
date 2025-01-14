from django.shortcuts import render
from django.core.paginator import Paginator

# Create your views here.
import pandas_datareader.data as web
import datetime
import yfinance as yf
from .models import Hmsp

# Override pandas datareader to use Yahoo Finance
yf.pdr_override()

def fetch_and_save_stock_data():
    start_date = datetime.datetime(2022, 5, 16)
    end_date = datetime.datetime.now()

    # Ambil data HMSP dari Yahoo Finance
    data = web.get_data_yahoo(['HMSP.JK'], start=start_date, end=end_date)
    # print(data.head())

    # Simpan ke database
    for date, row in data.iterrows():
        Hmsp.objects.update_or_create(
            date=date,
            defaults={
                'open_price': row['Open'],
                'high_price': row['High'],
                'low_price': row['Low'],
                'close_price': row['Close'],
                'volume': row['Volume']
            }
        )

# def stock_list(request):
#     stocks = Hmsp.objects.all().order_by('-date')
#     stock_count = stocks.count()
#     return render(request, 'html/stock_list.html', {'stocks': stocks, 'stock_count': stock_count})

def stock_list(request): 
    stocks = Hmsp.objects.all().order_by('-date') 
    paginator = Paginator(stocks, 100) # 10 stocks per page 
    
    page_number = request.GET.get('page') 
    page_obj = paginator.get_page(page_number)

    return render(request, 'html/stock_list.html', {'page_obj': page_obj, 'stock_count': stocks.count()})

def home(request):
    # stocks = Hmsp.objects.all().order_by('-date')
    # stock_count = stocks.count()
    return render(request, 'home/home.html')