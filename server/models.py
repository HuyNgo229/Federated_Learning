from django.db import models

class Client(models.Model):
    ip_address = models.CharField(verbose_name='ip_address', max_length=50)
    port = models.CharField(verbose_name='port', max_length=10)
    created_date = models.DateField(verbose_name='created_date', auto_now_add=True)
    name = models.CharField(verbose_name='name', max_length=50)
    