from django.db import models

class Video(models.Model):
    file = models.FileField(upload_to='input/')
