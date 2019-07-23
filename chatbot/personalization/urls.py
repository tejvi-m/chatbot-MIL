from django.contrib import admin
from . import views
from django.conf.urls import url
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

app_name='personalization'

urlpatterns = [
    url(r'^$',views.personalize, name="personalize"),
]


urlpatterns += staticfiles_urlpatterns()
