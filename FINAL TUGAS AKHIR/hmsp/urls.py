
from django.urls import path
from .views import stock_list , home

urlpatterns = [
    path('', stock_list, name='hmsp'),
    path('home/', home, name='home'),
]
