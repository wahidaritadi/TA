
from django.urls import path
from .views import stock_list

urlpatterns = [
    path('', stock_list, name='hmsp'),
]
