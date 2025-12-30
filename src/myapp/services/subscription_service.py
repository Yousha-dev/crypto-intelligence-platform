# subscription_service.py
from django.db import models
from django.utils import timezone
from datetime import datetime, timedelta
from myapp.models import Exchange, Users, Subscriptions, Subscriptionplans, APIUsage, UserExchangeCredentials
import logging

class SubscriptionService:
    """Service to handle trading subscription features and limits"""
    
    @classmethod
    def get_user_subscription(cls, user: Users) -> Subscriptions:
        """Get user's active subscription"""
        try:
            # First check for any active subscription
            subscription = Subscriptions.objects.select_related('subscriptionplanid').filter(
                userid=user,
                status='Active',
                isactive=1,
                isdeleted=0
            ).first()
            
            # If subscription exists, check if it's expired
            if subscription:
                if subscription.enddate < timezone.now().date():
                    # Mark as expired with proper audit fields
                    subscription.status = 'Expired'
                    subscription.updatedby = user.userid if hasattr(user, 'userid') else 1
                    subscription.updatedat = timezone.now()
                    subscription.save()
                    return cls._handle_expired_subscription(user, subscription)
                return subscription
            
            # No active subscription found, create default
            return cls._get_default_subscription(user)
            
        except Exception as e:
            # Log the error and raise it instead of calling _get_default_subscription again
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting user subscription for user {user.userid}: {str(e)}")
            raise Exception(f"Failed to get user subscription: {str(e)}")
    
    @classmethod
    def _handle_expired_subscription(cls, user: Users, expired_subscription: Subscriptions) -> Subscriptions:
        """Handle expired subscription - check if user had trial before"""
        try:
            # Check if user is eligible for trial
            is_eligible, message = cls.is_trial_eligible(user)
            
            if is_eligible:
                # User never had trial, give them one
                return cls._create_trial_subscription(user)
            else:
                # User already had trial, return the expired subscription
                # This will be handled in feature checks (all features disabled)
                return expired_subscription
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error handling expired subscription for user {user.userid}: {str(e)}")
            return expired_subscription
    
    @classmethod
    def _get_default_subscription(cls, user: Users) -> Subscriptions:
        """Create default subscription for new users only"""
        try:
            # Check if user ever had any subscription (including expired ones)
            existing_subscription = Subscriptions.objects.filter(
                userid=user,
                isdeleted=0
            ).first()
            
            if existing_subscription:
                # User had subscription before, check trial eligibility
                is_eligible, message = cls.is_trial_eligible(user)
                
                if not is_eligible:
                    # Had trial before, return existing subscription (even if expired)
                    return existing_subscription
                else:
                    # Had paid subscription but never had trial, can get trial
                    return cls._create_trial_subscription(user)
            
            # Completely new user - create trial subscription
            return cls._create_trial_subscription(user)
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create default subscription for user {user.userid}: {str(e)}")
            raise Exception(f"Failed to create default subscription: {str(e)}")
    
    @classmethod
    def _create_trial_subscription(cls, user: Users) -> Subscriptions:
        """Create trial subscription for eligible users"""
        try:
            # Double-check trial eligibility
            is_eligible, message = cls.is_trial_eligible(user)
            if not is_eligible:
                raise Exception(f"User not eligible for trial: {message}")
            
            # Get the free/trial plan
            trial_plan = Subscriptionplans.objects.filter(
                isactive=1,
                isdeleted=0,
                monthlyprice=0
            ).order_by('subscriptionplanid').first()
            
            if not trial_plan:
                # No free plan exists, get the cheapest plan for trial
                trial_plan = Subscriptionplans.objects.filter(
                    isactive=1,
                    isdeleted=0
                ).order_by('monthlyprice').first()
                
                if not trial_plan:
                    raise Exception("No subscription plans available in the system")
            
            # Create trial subscription
            subscription = Subscriptions.objects.create(
                userid=user,
                subscriptionplanid=trial_plan,
                billingfrequency='Monthly',
                startdate=timezone.now().date(),
                enddate=timezone.now().date() + timedelta(days=30),  # 30-day trial
                autorenew=0,
                status='Active',
                isactive=1,
                isdeleted=0,
                createdat=timezone.now(),
                createdby=user.userid if hasattr(user, 'userid') else 1
            )
            
            logger = logging.getLogger(__name__)
            logger.info(f"Created trial subscription for user {user.userid} with plan {trial_plan.name}")
            
            return subscription
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create trial subscription for user {user.userid}: {str(e)}")
            raise Exception(f"Failed to create trial subscription: {str(e)}")
    
    @classmethod
    def is_trial_eligible(cls, user: Users) -> tuple[bool, str]:
        """Check if user is eligible for a free trial"""
        try:
            # Check if user ever had a free trial (monthlyprice = 0)
            had_trial = Subscriptions.objects.filter(
                userid=user,
                subscriptionplanid__monthlyprice=0,
                isdeleted=0
            ).exists()
            
            if had_trial:
                return False, "User has already used their free trial"
            
            return True, "User is eligible for free trial"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking trial eligibility for user {user.userid}: {str(e)}")
            return False, f"Error checking trial eligibility: {str(e)}"
    
    @classmethod
    def is_subscription_valid(cls, user: Users) -> tuple[bool, str]:
        """Check if user has valid active subscription"""
        try:
            subscription = cls.get_user_subscription(user)
            
            if subscription.status != 'Active':
                return False, f"Subscription is {subscription.status}"
            
            if subscription.enddate < timezone.now().date():
                return False, "Subscription has expired"
            
            return True, "Subscription is valid"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking subscription validity for user {user.userid}: {str(e)}")
            return False, f"Error checking subscription: {str(e)}"
    
    @classmethod
    def get_subscription_features(cls, user: Users) -> dict:
        """Get user's subscription features from their subscription plan"""
        try:
            subscription = cls.get_user_subscription(user)
            plan = subscription.subscriptionplanid
            
            # Check if subscription is actually valid
            is_valid, validity_msg = cls.is_subscription_valid(user)
            
            return {
                'subscription_id': subscription.subscriptionid,
                'plan_id': plan.subscriptionplanid,
                'plan_name': plan.name,
                'plan_description': plan.description,
                'monthly_price': float(plan.monthlyprice) if plan.monthlyprice else 0,
                'yearly_price': float(plan.yearlyprice) if plan.yearlyprice else 0,
                'max_exchanges': plan.max_exchanges if is_valid else 0,
                'max_api_calls_per_hour': plan.max_api_calls_per_hour if is_valid else 0,
                'ai_predictions_enabled': plan.ai_predictions_enabled and is_valid,
                'advanced_indicators_enabled': plan.advanced_indicators_enabled and is_valid,
                'portfolio_tracking': plan.portfolio_tracking and is_valid,
                'trade_automation': plan.trade_automation and is_valid,
                'feature_details': plan.featuredetails,
                'subscription_status': subscription.status,
                'subscription_start_date': subscription.startdate,
                'subscription_end_date': subscription.enddate,
                'billing_frequency': subscription.billingfrequency,
                'auto_renew': bool(subscription.autorenew),
                'is_valid': is_valid,
                'validity_message': validity_msg
            }
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting subscription features for user {user.userid}: {str(e)}")
            return {
                'error': str(e),
                'max_exchanges': 0,
                'max_api_calls_per_hour': 0,
                'ai_predictions_enabled': False,
                'advanced_indicators_enabled': False,
                'portfolio_tracking': False,
                'trade_automation': False,
                'is_valid': False,
                'validity_message': f'Error: {str(e)}'
            }
    
    @classmethod
    def check_api_limit(cls, user: Users) -> tuple[bool, dict]:
        """Check if user has reached API limit"""
        try:
            # Check subscription validity first
            is_valid, validity_msg = cls.is_subscription_valid(user)
            
            if not is_valid:
                return False, {
                    'error': 'Subscription invalid',
                    'message': validity_msg,
                    'used_calls': 0,
                    'max_calls': 0,
                    'remaining_calls': 0
                }
            
            features = cls.get_subscription_features(user)
            max_calls = features['max_api_calls_per_hour']
            
            # Check API usage in the last hour
            one_hour_ago = timezone.now() - timedelta(hours=1)
            usage_count = APIUsage.objects.filter(
                user=user,
                timestamp__gte=one_hour_ago,
                isactive=1,
                isdeleted=0
            ).aggregate(
                total=models.Sum('request_count')
            )['total'] or 0
            
            remaining_calls = max_calls - usage_count
            can_make_call = usage_count < max_calls
            
            return can_make_call, {
                'used_calls': usage_count,
                'max_calls': max_calls,
                'remaining_calls': max(0, remaining_calls),
                'reset_time': one_hour_ago + timedelta(hours=1)
            }
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking API limit for user {user.userid}: {str(e)}")
            return False, {'error': str(e)}
    
    @classmethod
    def can_use_ai_predictions(cls, user: Users) -> tuple[bool, str]:
        """Check if user can access AI predictions"""
        try:
            is_valid, validity_msg = cls.is_subscription_valid(user)
            
            if not is_valid:
                return False, validity_msg
            
            features = cls.get_subscription_features(user)
            
            if not features['ai_predictions_enabled']:
                return False, f"AI predictions not available in {features['plan_name']} plan"
            
            return True, "AI predictions available"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking AI predictions access for user {user.userid}: {str(e)}")
            return False, f"Error checking AI predictions access: {str(e)}"
    
    @classmethod
    def can_use_advanced_indicators(cls, user: Users) -> tuple[bool, str]:
        """Check if user can access advanced indicators"""
        try:
            is_valid, validity_msg = cls.is_subscription_valid(user)
            
            if not is_valid:
                return False, validity_msg
            
            features = cls.get_subscription_features(user)
            
            if not features['advanced_indicators_enabled']:
                return False, f"Advanced indicators not available in {features['plan_name']} plan"
            
            return True, "Advanced indicators available"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking advanced indicators access for user {user.userid}: {str(e)}")
            return False, f"Error checking advanced indicators access: {str(e)}"
    
    @classmethod
    def can_use_portfolio_tracking(cls, user: Users) -> tuple[bool, str]:
        """Check if user can access portfolio tracking"""
        try:
            is_valid, validity_msg = cls.is_subscription_valid(user)
            
            if not is_valid:
                return False, validity_msg
            
            features = cls.get_subscription_features(user)
            
            if not features['portfolio_tracking']:
                return False, f"Portfolio tracking not available in {features['plan_name']} plan"
            
            return True, "Portfolio tracking available"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking portfolio tracking access for user {user.userid}: {str(e)}")
            return False, f"Error checking portfolio tracking access: {str(e)}"
    
    @classmethod
    def can_use_trade_automation(cls, user: Users) -> tuple[bool, str]:
        """Check if user can access trade automation"""
        try:
            is_valid, validity_msg = cls.is_subscription_valid(user)
            
            if not is_valid:
                return False, validity_msg
            
            features = cls.get_subscription_features(user)
            
            if not features['trade_automation']:
                return False, f"Trade automation not available in {features['plan_name']} plan"
            
            return True, "Trade automation available"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking trade automation access for user {user.userid}: {str(e)}")
            return False, f"Error checking trade automation access: {str(e)}"
    
    @classmethod
    def check_exchange_limit(cls, user: Users, current_exchanges_count: int = 0) -> tuple[bool, dict]:
        """Check if user can add more exchanges"""
        try:
            is_valid, validity_msg = cls.is_subscription_valid(user)
            
            if not is_valid:
                return False, {
                    'error': validity_msg,
                    'current_exchanges': 0,
                    'max_exchanges': 0,
                    'remaining_exchanges': 0
                }
            
            features = cls.get_subscription_features(user)
            max_exchanges = features['max_exchanges']
            
            if current_exchanges_count == 0:
                current_exchanges_count = UserExchangeCredentials.objects.filter(
                    user=user,
                    isactive=1,
                    isdeleted=0
                ).count()
            
            can_add_exchange = current_exchanges_count < max_exchanges
            remaining_exchanges = max_exchanges - current_exchanges_count
            
            return can_add_exchange, {
                'current_exchanges': current_exchanges_count,
                'max_exchanges': max_exchanges,
                'remaining_exchanges': max(0, remaining_exchanges)
            }
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking exchange limit for user {user.userid}: {str(e)}")
            return False, {'error': str(e)}
    
    @classmethod
    def record_api_usage(cls, user: Users, exchange, endpoint_type: str, request_count: int = 1):
        """Record API usage for tracking and billing"""
        try:
            # Handle both Exchange objects and exchange names
            if isinstance(exchange, str):
                exchange_obj = Exchange.objects.filter(name=exchange.lower(), isactive=1, isdeleted=0).first()
                if not exchange_obj:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Exchange {exchange} not found for API usage recording")
                    return
            else: 
                exchange_obj = exchange
                
            APIUsage.objects.create(
                user=user,
                exchange=exchange_obj,
                endpoint_type=endpoint_type,
                request_count=request_count,
                timestamp=timezone.now(),
                isactive=1,
                isdeleted=0
            )
        except Exception as e:
            # Log error but don't fail the API call
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to record API usage for user {user.userid}: {str(e)}")
    
    @classmethod
    def get_subscription_stats(cls, user: Users) -> dict:
        """Get comprehensive subscription statistics"""
        try:
            features = cls.get_subscription_features(user)
            api_check = cls.check_api_limit(user)
            exchange_check = cls.check_exchange_limit(user)
            ai_check = cls.can_use_ai_predictions(user)
            
            # Get recent API usage
            one_day_ago = timezone.now() - timedelta(days=1)
            daily_usage = APIUsage.objects.filter(
                user=user,
                timestamp__gte=one_day_ago,
                isactive=1,
                isdeleted=0
            ).aggregate(
                total=models.Sum('request_count')
            )['total'] or 0
            
            return {
                'subscription': features,
                'api_limits': api_check[1] if len(api_check) > 1 else {},
                'exchange_limits': exchange_check[1] if len(exchange_check) > 1 else {},
                'ai_predictions_available': ai_check[0],
                'daily_api_usage': daily_usage,
                'features_available': {
                    'ai_predictions': ai_check[0],
                    'advanced_indicators': cls.can_use_advanced_indicators(user)[0],
                    'portfolio_tracking': cls.can_use_portfolio_tracking(user)[0],
                    'trade_automation': cls.can_use_trade_automation(user)[0]
                }
            }
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting subscription stats for user {user.userid}: {str(e)}")
            return {'error': str(e)}
    
    @classmethod
    def change_user_subscription_plan(cls, user: Users, new_plan_id: int) -> tuple[bool, str]:
        """Change user to a different subscription plan"""
        try:
            # Validate new plan exists
            new_plan = Subscriptionplans.objects.get(
                subscriptionplanid=new_plan_id,
                isactive=1,
                isdeleted=0
            )
            
            # Get current subscription
            current_subscription = cls.get_user_subscription(user)
            
            # Don't allow changing to the same plan
            if current_subscription.subscriptionplanid.subscriptionplanid == new_plan_id:
                return False, "Already subscribed to this plan"
            
            # Update the subscription plan
            current_subscription.subscriptionplanid = new_plan
            current_subscription.updatedby = user.userid if hasattr(user, 'userid') else 1
            current_subscription.updatedat = timezone.now()
            
            # Handle subscription duration based on plan type and billing frequency
            if new_plan.monthlyprice > 0:
                # Paid plan - extend subscription based on billing frequency
                if current_subscription.billingfrequency == 'Yearly':
                    current_subscription.enddate = timezone.now().date() + timedelta(days=365)
                else:
                    current_subscription.enddate = timezone.now().date() + timedelta(days=30)
                current_subscription.status = 'Active'
            else:
                # Free plan - set trial period
                current_subscription.enddate = timezone.now().date() + timedelta(days=30)
                current_subscription.status = 'Active'
            
            current_subscription.save()
            
            logger = logging.getLogger(__name__)
            logger.info(f"Changed subscription for user {user.userid} to plan {new_plan.name}")
            
            return True, f"Successfully changed to {new_plan.name}"
            
        except Subscriptionplans.DoesNotExist:
            return False, "Subscription plan not found"
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to change subscription for user {user.userid}: {str(e)}")
            return False, f"Failed to change subscription: {str(e)}"
    
    @classmethod
    def get_available_plans(cls) -> list:
        """Get all available subscription plans ordered by price"""
        try:
            plans = Subscriptionplans.objects.filter(
                isactive=1,
                isdeleted=0
            ).order_by('monthlyprice')
            
            return [{
                'plan_id': plan.subscriptionplanid,
                'name': plan.name,
                'description': plan.description,
                'monthly_price': float(plan.monthlyprice) if plan.monthlyprice else 0,
                'yearly_price': float(plan.yearlyprice) if plan.yearlyprice else 0,
                'max_exchanges': plan.max_exchanges,
                'max_api_calls_per_hour': plan.max_api_calls_per_hour,
                'ai_predictions_enabled': plan.ai_predictions_enabled,
                'advanced_indicators_enabled': plan.advanced_indicators_enabled,
                'portfolio_tracking': plan.portfolio_tracking,
                'trade_automation': plan.trade_automation,
                'feature_details': plan.featuredetails
            } for plan in plans]
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting available plans: {str(e)}")
            return []
    
    @classmethod
    def is_plan_upgrade(cls, user: Users, new_plan_id: int) -> tuple[bool, dict]:
        """Check if the new plan is an upgrade, downgrade, or same level"""
        try:
            current_features = cls.get_subscription_features(user)
            new_plan = Subscriptionplans.objects.get(
                subscriptionplanid=new_plan_id,
                isactive=1,
                isdeleted=0
            )
            
            current_price = current_features['monthly_price']
            new_price = float(new_plan.monthlyprice) if new_plan.monthlyprice else 0
            
            if new_price > current_price:
                change_type = "upgrade"
            elif new_price < current_price:
                change_type = "downgrade"
            else:
                change_type = "same_level"
            
            return True, {
                'change_type': change_type,
                'current_plan': current_features['plan_name'],
                'new_plan': new_plan.name,
                'current_price': current_price,
                'new_price': new_price,
                'price_difference': new_price - current_price
            }
            
        except Subscriptionplans.DoesNotExist:
            return False, {'error': 'New plan not found'}
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking plan upgrade for user {user.userid}: {str(e)}")
            return False, {'error': str(e)}
    
    @classmethod
    def get_or_create_subscription(cls, user: Users) -> Subscriptions:
        """Get or create subscription for user - alias for get_user_subscription"""
        return cls.get_user_subscription(user)
    
    @classmethod
    def extend_subscription(cls, user: Users, days: int) -> tuple[bool, str]:
        """Extend user's current subscription by specified days"""
        try:
            subscription = cls.get_user_subscription(user)
            
            # Extend the end date
            subscription.enddate = subscription.enddate + timedelta(days=days)
            subscription.updatedby = user.userid if hasattr(user, 'userid') else 1
            subscription.updatedat = timezone.now()
            subscription.save()
            
            logger = logging.getLogger(__name__)
            logger.info(f"Extended subscription for user {user.userid} by {days} days")
            
            return True, f"Subscription extended by {days} days until {subscription.enddate}"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to extend subscription for user {user.userid}: {str(e)}")
            return False, f"Failed to extend subscription: {str(e)}"
     
    @classmethod
    def cancel_subscription(cls, user: Users) -> tuple[bool, str]:
        """Cancel user's subscription (mark as inactive)"""
        try:
            subscription = cls.get_user_subscription(user)
             
            # Mark subscription as cancelled
            subscription.status = 'Cancelled'
            subscription.autorenew = 0
            subscription.updatedby = user.userid if hasattr(user, 'userid') else 1
            subscription.updatedat = timezone.now()
            subscription.save()
            
            logger = logging.getLogger(__name__)
            logger.info(f"Cancelled subscription for user {user.userid}")
            
            return True, "Subscription cancelled successfully"
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to cancel subscription for user {user.userid}: {str(e)}")
            return False, f"Failed to cancel subscription: {str(e)}"