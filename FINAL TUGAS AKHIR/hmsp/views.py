from django.shortcuts import render
from django.core.paginator import Paginator

# Create your views here.
import pandas_datareader.data as web
import datetime
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from .models import Hmsp

# Override pandas datareader to use Yahoo Finance
yf.pdr_override()

def fetch_and_save_stock_data():
    start_date = datetime.datetime(2015, 5, 18)
    end_date = datetime.datetime.now()

    # Ambil data HMSP dari Yahoo Finance
    # data = web.get_data_yahoo(['HMSP.JK'], start=start_date, end=end_date)
    data = web.get_data_yahoo('HMSP.JK', start=start_date, end=end_date)

    data.reset_index(inplace=True)
    print(data.head())
    # Proses data seperti di notebook
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    features = data[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day']]
    target = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi untuk hari berikutnya
    last_row = data.iloc[-1].copy()
    if last_row['Day'] > 31:
        last_row['Day'] = 1
        last_row['Month'] += 1
    if last_row['Month'] > 12:
        last_row['Month'] = 1
        last_row['Year'] += 1

    last_row_df = pd.DataFrame([last_row])
    next_day_prediction = model.predict(last_row_df[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day']])
    next_day_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
    predicted_data = pd.DataFrame({'Date': [next_day_date], 'Predicted_Close': [next_day_prediction[0]]})

    # Simpan ke database
    for date, row in data.iterrows():
        Hmsp.objects.update_or_create(
            date=row['Date'],
            defaults={
                'open_price': row['Open'],
                'high_price': row['High'],
                'low_price': row['Low'],
                'close_price': row['Close'],
                'volume': row['Volume']
            }
        )

def stock_list(request):
    stocks = Hmsp.objects.all().order_by('-date')
    stocksChart = Hmsp.objects.all().order_by('date')
    paginator = Paginator(stocks, 100)  # 10 stocks per page

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Ambil data untuk grafik
    chart_data = list(stocksChart.values('date', 'close_price'))
    dates = [data['date'].strftime('%Y-%m-%d') for data in chart_data]
    close_prices = [data['close_price'] for data in chart_data]

    # Data Prediksi
    start_date = datetime.datetime(2015, 5, 16)
    end_date = datetime.datetime.now()
    hmsp = web.get_data_yahoo('HMSP.JK', start=start_date, end=end_date)
    hmsp['Date'] = pd.to_datetime(hmsp.index)
    hmsp['Year'] = hmsp['Date'].dt.year
    hmsp['Month'] = hmsp['Date'].dt.month
    hmsp['Day'] = hmsp['Date'].dt.day
    features = hmsp[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day']]
    target = hmsp['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi untuk hari berikutnya
    last_row = hmsp.iloc[-1].copy()
    if last_row['Day'] > 31:
        last_row['Day'] = 1
        last_row['Month'] += 1
    if last_row['Month'] > 12:
        last_row['Month'] = 1
        last_row['Year'] += 1
    last_row_df = pd.DataFrame([last_row])
    next_day_prediction = model.predict(last_row_df[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day']])
    next_day_date = hmsp['Date'].iloc[-1] + pd.Timedelta(days=1)
    predicted_data = pd.DataFrame({'Date': [next_day_date], 'Predicted_Close': [next_day_prediction[0]]})

    # Data Prediksi untuk Grafik
    predicted_dates = [date.strftime('%Y-%m-%d') for date in predicted_data['Date']]
    predicted_close_prices = predicted_data['Predicted_Close'].tolist()

    return render(request, 'html/stock_list.html', {
        'page_obj': page_obj,
        'stock_count': stocks.count(),
        'dates': dates,
        'close_prices': close_prices,
        'predicted_dates': predicted_dates,
        'predicted_close_prices': predicted_close_prices
    })

def home(request):
    return render(request, 'home/home.html')

