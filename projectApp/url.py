from django.contrib import admin
from django.urls import path , include 
from . import views
from django.conf import settings
from django.conf.urls.static import static
#UrlConfiguration

app_name = 'pages'

urlpatterns = [
    path('Predict/', views.home, name='PredictionPage'),
    path('contact/', views.contact ,name='contact'),
    path('blog/', views.blog ,name='blog'),
    path('team/', views.team ,name='team'),
    path('home/', views.index ,name='index'),

    path('Lung/', views.lung ,name='lung'),
    path('Loading/', views.Load ,name='Load'),
    path('Signature/', views.signpro ,name='Signature'),
    path('Choix2/', views.Choice ,name='Choice'),
    path('ChoiceNew/', views.ChoiceNew, name='ChoiceNew'),
    path('Reviews/', views.Reviews, name='Reviews'),
    path('SkinDisease/', views.SkinDisease, name='SkinDisease'),
    path('about/', views.about, name='about'),
    path('LungCancerPage/', views.LungCancerPage, name='LungCancerPage'),
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('activate/<uidb64>/<token>', views.activate, name='activate'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
]
