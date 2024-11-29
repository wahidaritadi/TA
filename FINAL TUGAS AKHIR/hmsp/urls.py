
from django.urls import path
from .views import callStock_list , home , predictData

urlpatterns = [
    path('', home, name='home'),
    path('DataHMSP', callStock_list, name='hmsp'),
    path('PrediksiNilaiHSMP', predictData, name='PrediksiNilaiHSMP'),
]
