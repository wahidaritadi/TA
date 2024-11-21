# myapp/views.py
from django.shortcuts import render

def home(request):
    # stocks = Hmsp.objects.all().order_by('-date')
    # stock_count = stocks.count()
    return render(request, 'home/home.html')
