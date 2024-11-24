
from django.urls import path
from .views import callStock_list , home

urlpatterns = [
    path('', callStock_list, name='hmsp'),
    path('home/', home, name='home'),
]
