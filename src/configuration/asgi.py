"""
ASGI config for this project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""
import os
import django
from django.core.asgi import get_asgi_application

# Set Django settings FIRST
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'configuration.settings')

# Initialize Django BEFORE importing anything that uses models
django.setup()

# Now import channels and routing (after Django is setup)
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from myapp.routing import websocket_urlpatterns

# Create ASGI application
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})