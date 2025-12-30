#middleware.py

from rest_framework.exceptions import AuthenticationFailed
from myapp.authentication import CustomJWTAuthentication
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.http import JsonResponse
from myapp.services.subscription_service import SubscriptionService
import logging
import json
 
logger = logging.getLogger(__name__)

class AddCustomFieldsToHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response  # Initialize the response handler

    def __call__(self, request):
        # Check if the Authorization header exists
        if "Authorization" in request.headers:
            auth = CustomJWTAuthentication()
            try:
                _, token = auth.authenticate(request)
                if token:
                    # Add custom fields to the request META
                    request.META["HTTP_X_USER_ID"] = token.get("user_id")
                    request.META["HTTP_X_ROLE"] = token.get("role")
                    print("Token claims:", token)  # Debug: Log token claims
            except AuthenticationFailed as e:
                # Debug: Log authentication failure 
                print("Authentication failed:", e)

        # Call the next middleware or the view
        response = self.get_response(request)
        return response


class AuthenticateUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if "Authorization" in request.headers:
            auth = CustomJWTAuthentication()
            try:
                print("abacsac")
                _, token = auth.authenticate(request)
                if token:
                    request.user_id = token.get("user_id")
                    request.role = token.get("role")
            except AuthenticationFailed as e:
                print(f"Authentication failed: {e}")

        response = self.get_response(request)
        return response


class UserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        auth = JWTAuthentication()
        try:
            auth_result = auth.authenticate(request)
            if auth_result:
                user, token = auth_result
                request.user_id = token.get('user_id')
            else:
                request.user_id = None
        except AuthenticationFailed as e:
            print(f"Authentication failed: {e}")
            request.user_id = None

        response = self.get_response(request)
        return response


class TradingMiddleware:
    """Middleware to handle trading rate limiting and subscription checks"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Only apply to trading endpoints
        if request.path.startswith('/api/core/trading/') or request.path.startswith('/api/v1/core/trading/'):
            if hasattr(request, 'user') and request.user.is_authenticated:
                try:
                    # Initialize subscription if needed
                    SubscriptionService.get_or_create_subscription(request.user)
                    
                    # Add user features to request for easy access
                    request.trading_features = SubscriptionService.get_subscription_features(request.user)
                    
                except Exception as e:
                    logger.error(f"Error initializing subscription for {request.user.email}: {e}")
                    # Don't block request, just log error
                    request.trading_features = {
                        'plan_name': 'Error',
                        'ai_predictions_enabled': False,
                        'advanced_indicators_enabled': False,
                        'portfolio_tracking': False,
                        'trade_automation': False,
                        'max_api_calls': 0,
                        'max_exchanges': 0
                    }
            elif request.path.startswith('/api/core/trading/') or request.path.startswith('/api/v1/core/trading/'):
                # For anonymous users on trading endpoints, set default features
                request.trading_features = {
                    'plan_name': 'Anonymous',
                    'ai_predictions_enabled': False,
                    'advanced_indicators_enabled': False,
                    'portfolio_tracking': False,
                    'trade_automation': False,
                    'max_api_calls': 0,
                    'max_exchanges': 0
                }
        
        response = self.get_response(request)

        # Add subscription info to API responses if it's a trading endpoint
        if (request.path.startswith('/api/core/trading/') or request.path.startswith('/api/v1/core/trading/')) and \
           hasattr(request, 'trading_features'):
            
            try:
                # Only add to successful JSON responses
                if (response.status_code == 200 and 
                    response.get('Content-Type', '').startswith('application/json') and
                    hasattr(request, 'user') and request.user.is_authenticated):
                    
                    try:
                        # Parse existing response data
                        response_data = json.loads(response.content.decode('utf-8'))
                        
                        # Add subscription info if response is a dict
                        if isinstance(response_data, dict):
                            response_data['_subscription_plan'] = request.trading_features.get('plan_name', 'Unknown')
                            
                            # Re-encode response
                            response.content = json.dumps(response_data).encode('utf-8')
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # If we can't parse the response, just skip adding subscription info
                        pass
                        
            except Exception as e:
                logger.error(f"Error adding subscription info to response: {e}")
        
        return response


class APIRateLimitMiddleware:
    """Rate limiting middleware for API endpoints"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Apply rate limiting to API endpoints for authenticated users
        if (request.path.startswith('/api/core/trading/') and 
            hasattr(request, 'user') and 
            request.user.is_authenticated):
            
            try:
                # Check if user can make API calls
                can_use_api, message = SubscriptionService.can_use_api(request.user)
                if not can_use_api:
                    return JsonResponse({
                        'error': 'API rate limit exceeded',
                        'message': message,
                        'status': 429
                    }, status=429)
                    
            except Exception as e:
                logger.error(f"Error checking API rate limit for {request.user.email}: {e}")
                # Don't block request on error, just log it
        
        response = self.get_response(request)
        return response