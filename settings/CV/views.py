from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *


def index(request):
    pass

def takeImg(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        print(form.path)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = ImageForm()
    return render(request, 'CV/takeCard.html', {'form' : form})
  