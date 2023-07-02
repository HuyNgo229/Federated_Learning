from django.urls import path
from . import views

app_name = 'server'

urlpatterns = [
    path('index', views.Home.as_view(), name='index'),

    # client action
    path('login', views.Login.as_view(), name='login'),
    path('add/client', views.AddClient, name='add_client'),
    path('edit/client', views.EditClient, name='edit_client'),
    path('detail/client', views.DetailClient, name='detail_client'),
    path('remove/client', views.RemoveClient, name='remove_client'),

    # train
    
]
