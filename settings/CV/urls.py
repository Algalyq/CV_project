from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('take/', views.takeImg, name='take'),
    path('recognition/', views.recognition, name='recog'),
    path('profile/<str:data>', views.profile, name='profile'),
]
