from django.db import models
class Doctor(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)

    class Meta:
        app_label = 'projectApp'
# Create your models here.
