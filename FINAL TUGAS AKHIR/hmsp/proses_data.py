from django.shortcuts import render
from django.core.paginator import Paginator
from django.db import models
from django.utils import timezone
import pytz

# Create your views here.
import pandas_datareader.data as web
import datetime
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .models import Hmsp, Prediction, LGPrediction

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
    

    return render(request, 'html/stock_list.html', {
        'page_obj': page_obj,
        'stock_count': stocks.count(),
        'dates': dates,
        'close_prices': close_prices
    })
# Function to predict for a single day
def predict_for_single_day(model, last_row):
    last_row['Day'] += 1
    if last_row['Day'] > 31:
        last_row['Day'] = 1
        last_row['Month'] += 1
    if last_row['Month'] > 12:
        last_row['Month'] = 1
        last_row['Year'] += 1
    return model.predict(pd.DataFrame([last_row])[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day']])[0]

def predict_stock_price(request):
    if request.method == 'POST':
        predictStart_date = request.POST['start_date']
        predictEnd_date = request.POST['end_date']
        predictionGenerateTime = timezone.now().astimezone(pytz.timezone('Asia/Jakarta'))
        
        # Get the maximum predictionRef and increment by 1 
        max_ref = Prediction.objects.aggregate(max_ref=models.Max('predictionRef'))['max_ref'] 
        if max_ref is None: 
            max_ref = 0 
        predictionRef = max_ref + 1

        # Convert input date to datetime
        predictStart_date = datetime.datetime.strptime(predictStart_date, '%Y-%m-%d')
        predictEnd_date = datetime.datetime.strptime(predictEnd_date, '%Y-%m-%d')
        
        # Data Prediksi
        start_date = datetime.datetime(2005, 5, 16)
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

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        accuracy = model.score(X_test, y_test)
        
        mse = round(mse, 2)
        rmse = round(rmse, 2)
        accuracy = round(accuracy * 100, 2)
        r2 = round(r2 * 100, 2)

        # Generate predictions for the date range
        prediction_dates = pd.date_range(start=predictStart_date, end=predictEnd_date)
        prediction_results = []
        last_row = hmsp.iloc[-1].copy()

        for date in prediction_dates:
            last_row['Year'] = date.year
            last_row['Month'] = date.month
            last_row['Day'] = date.day
            prediction = predict_for_single_day(model, last_row)
            prediction_results.append(float(prediction))

            # Save prediction to database 
            Prediction.objects.create( 
                predictionRef=predictionRef, 
                date=date, 
                predicted_close_price=prediction,
            )

            # Update last_row for next prediction
            last_row['Open'] = prediction
            last_row['High'] = prediction
            last_row['Low'] = prediction
            last_row['Close'] = prediction
            last_row['Volume'] = hmsp['Volume'].iloc[-1]

        # Save prediction log to LGPrediction 
        LGPrediction.objects.create( 
            predictionRef=predictionRef,
            predictionDateFrom=predictStart_date,
            predictionDateTo=predictEnd_date,
            inputTime=predictionGenerateTime
        )

        context = {
            'dates': [date.strftime('%Y-%m-%d') for date in prediction_dates],
            'predictions': prediction_results,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy
            # 'predictions': prediction_results.tolist()
        }
        
        return render(request, 'html/predictionStrockPrice.html', context)

    return render(request, 'html/predictionStrockPrice.html')

def prediction_history(request):
    predictions = LGPrediction.objects.all().order_by('-inputTime')
    context = {
        'predictions': predictions
    }
    return render(request, 'html/History.html', context)

def prediction_detail(request, predictionRef): 
    predictions = Prediction.objects.filter(predictionRef=predictionRef).order_by('date') 
    context = { 
        'predictions': predictions, 
        'predictionRef': predictionRef
    }

    return render(request, 'html/HistoryDetail.html', context)