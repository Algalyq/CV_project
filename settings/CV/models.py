from django.db import models

from django.contrib.auth import get_user_model
User=get_user_model()

class Images(models.Model):
    id_img = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to ='uploads/')
