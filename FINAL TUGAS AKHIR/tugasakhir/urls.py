"""
URL configuration for tugasakhir project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import view
# from hmsp.urls import urls
from hmsp.views import callStock_list , home , predictStockPrice

urlpatterns = [
    # path('', include('hmsp.urls')),
    # path('admin/', admin.site.urls),9
    path('', home, name='home'),
    path('DataHMSP', callStock_list, name='hmsp'),
    path('PrediksiNilaiHSMP', predictStockPrice, name='PrediksiNilaiHSMP'),
]
