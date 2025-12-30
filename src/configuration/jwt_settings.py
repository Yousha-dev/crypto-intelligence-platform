# jwt_settings.py

from datetime import timedelta, datetime
from rest_framework_simplejwt.settings import api_settings

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=10),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=15),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_CLAIM': 'user_id',
    'USER_ROLE_CLAIM': 'role',
}

def custom_jwt_payload_handler(user):
    exp = datetime.utcnow() + api_settings.ACCESS_TOKEN_LIFETIME
    payload = {
        'user_id': user.id,  # Replace with your custom user field
        'role': user.role,  # Add role to payload
        'exp': int(exp.timestamp()),
    }
    return payload

api_settings.JWT_PAYLOAD_HANDLER = custom_jwt_payload_handler

SWAGGER_SETTINGS = {
    'SCHEME': ['https', 'http'],
    'SERVERS': [
        {'url': 'https://untutelar-anaya-noninferentially.ngrok-free.dev', 'description': 'Ngrok Public Access'},  
        {'url': 'http://localhost:8000', 'description': 'Local Development'},
        {'url': 'http://127.0.0.1:8000', 'description': 'Local Alternative'}
    ],
    'SECURITY_DEFINITIONS': {
        'User Authentication': {
            'type': 'apiKey',
            'name': 'Authorization',
            'in': 'header',
            'description': (
                "Enter your Bearer token in the format: Bearer <token>.\n\n"
                "Alternatively, enter the following fields manually to test:\n"
                "- `user_id`: User's ID.\n"
                "- `role`: User's role."
            ),
            'x-fields': {
                'user_id': {
                    'type': 'string',
                    'description': 'User ID',
                },
                'role': {
                    'type': 'string',
                    'description': 'User Role',
                },
            },
        }
    },
    'USE_SESSION_AUTH': False,
}
