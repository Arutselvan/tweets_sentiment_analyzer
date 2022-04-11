from django.urls import path
from .consumers import TweetConsumer

ws_urlpatterns = [
    path('ws/stream/', TweetConsumer.as_asgi())
]