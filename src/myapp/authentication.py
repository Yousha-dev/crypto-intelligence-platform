# authentication.py

import logging
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed
from .models import Users
 
logger = logging.getLogger(__name__)

class CustomJWTAuthentication(JWTAuthentication):
    def get_user(self, validated_token):
        user_id = validated_token.get('user_id')
        if not user_id:
            raise AuthenticationFailed("User ID not found in token", code="user_id_missing")

        try:
            user = Users.objects.get(userid=user_id, isactive=1, isdeleted=0)  
        except Users.DoesNotExist:
            raise AuthenticationFailed("User not found", code="user_not_found")

        return user

    def authenticate(self, request):
        logger.debug("Starting authentication process.")
        
        header = self.get_header(request)
        if header is None:
            logger.debug("No Authorization header found.")
            return None

        raw_token = self.get_raw_token(header)
        if raw_token is None:
            logger.debug("No raw token found in the Authorization header.")
            return None

        try:
            validated_token = self.get_validated_token(raw_token)
            logger.debug("Token successfully validated. Claims: %s", validated_token)
            
            # Attach claims to the request object directly
            request.user_id = validated_token.get('user_id')
            request.role = validated_token.get('role')

            # Return None for user to avoid relying on Django's user model
            return None, validated_token
        except Exception as e:
            logger.error("Authentication failed: %s", str(e))
            raise AuthenticationFailed(str(e))