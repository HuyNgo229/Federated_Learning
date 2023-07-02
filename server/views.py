import json
from django.http import HttpResponse
from rest_framework import status
from django.shortcuts import render
from django.views.generic import TemplateView
from rest_framework.views import APIView
from . import models

class Home(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context
    
class Login(TemplateView):
    template_name = "auth/login.html"

def AddClient(request):
    if request.method == 'GET':
        return render(request, "client/add.html")

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'add_new_client':
            data = json.loads(request.POST.get('params'))
            ip_address = data.get('ip_address')
            port = data.get('port')
            name = data.get('name')
            created = models.Client.objects.create(
                ip_address = ip_address,
                port = port,
                name = name
            )
            if created:        
                return HttpResponse('create client success', status = status.HTTP_201_CREATED)
            else:
                return HttpResponse('fail', status = status.HTTP_400_BAD_REQUEST)
                

    return HttpResponse('fail', status = status.HTTP_400_BAD_REQUEST)    
    
def EditClient(request):
    pass

def DetailClient(request):
    pass

def RemoveClient(request):
    pass

class TrainingClient(APIView):
    def post():
        