from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('take/', views.takeImg, name='take'),
    path('recog/', views.Recog, name='recog'),
    path('tema/<str:data>', views.tema, name='tema'),
]
