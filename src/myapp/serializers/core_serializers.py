from decimal import Decimal
from rest_framework import serializers
from django.utils import timezone
from django.core.validators import validate_email
from rest_framework import serializers
from django.utils import timezone
from django.conf import settings
from cryptography.fernet import Fernet
import os
from myapp.models import (
    EVENTS_CATEGORY_CHOICES, EVENTS_FREQUENCY_CHOICES, EVENTS_TYPE_CHOICES, NOTIFICATIONS_TYPE_CHOICES, Events, Notifications, Reminders,
    Exchange, TradingPair, UserExchangeCredentials, APIUsage, UserTrade, UserPortfolio, Users
)
from myapp.serializers.admin_serializers import SubscriptionSerializer

    
class NotificationSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)  # Default to True (1)
    isdeleted = serializers.IntegerField(default=0)  # Default to False (0)
    isread = serializers.IntegerField(default=0)    # Default to False (0)
    createdat = serializers.DateTimeField(read_only=True)  # Add this line
    updatedat = serializers.DateTimeField(read_only=True)  # Add this line
    type = serializers.ChoiceField(
        choices=NOTIFICATIONS_TYPE_CHOICES,
        error_messages={
            'invalid_choice': 'Invalid notification type. Valid choices are: {}'.format(
                ', '.join([choice[0] for choice in NOTIFICATIONS_TYPE_CHOICES])
            )
        }
    )

    class Meta:
        model = Notifications
        fields = [
            'notificationid', 'userid', 'title', 'message', 'type',
            'daysuntilexpiry',
            'isread', 'isactive', 'isdeleted', 'createdby', 'updatedby', 'createdat', 'updatedat'
        ]

    def create(self, validated_data):
        validated_data['createdat'] = timezone.now()
        return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        return super().update(instance, validated_data)
    
    def validate(self, data):
        """
        Custom validation for notification data.
        - Ensure type is valid
        - Validate daysuntilexpiry is positive if provided
        """
        # Validate daysuntilexpiry if provided
        if 'daysuntilexpiry' in data and data['daysuntilexpiry'] is not None:
            if data['daysuntilexpiry'] < 0:
                raise serializers.ValidationError({
                    "daysuntilexpiry": "Days until expiry must be a positive number."
                })

        # Ensure at least one related entity is provided
        # if not self.instance:  # create operation
        #     if ('extensionsubscriptionid' not in data or data['extensionsubscriptionid'] is None) and \
        #     ('contractid' not in data or data['contractid'] is None):
        #         raise serializers.ValidationError(
        #             "Either extensionsubscriptionid or contractid must be provided."
        #         )

        return data
    
class ReminderSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)  # Default to True (1)
    isdeleted = serializers.IntegerField(default=0)  # Default to False (0)
    timestamp = serializers.DateTimeField(required=True)  # Ensure timestamp is provided

    class Meta:
        model = Reminders
        fields = [
            'reminderid', 'userid', 'note',
            'timestamp', 'isactive', 'isdeleted', 'createdby',
            'updatedby'
        ]
    
    def create(self, validated_data):
        validated_data['createdat'] = timezone.now()
        return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        return super().update(instance, validated_data)

    def validate(self, data):
        # Validate timestamp
        if 'timestamp' in data:
            if data['timestamp'] < timezone.now():
                raise serializers.ValidationError({
                    "timestamp": "Reminder timestamp cannot be in the past."
                })

        # Validate note length
        if 'note' in data and len(data['note']) > 2000:  # Adjust max length as needed
            raise serializers.ValidationError({
                "note": "Note cannot exceed 2000 characters."
            })


        return data
    
class EventSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    type = serializers.ChoiceField(choices=EVENTS_TYPE_CHOICES)
    category = serializers.ChoiceField(choices=EVENTS_CATEGORY_CHOICES)
    frequency = serializers.ChoiceField(choices=EVENTS_FREQUENCY_CHOICES, allow_null=True, required=False)

    class Meta:
        model = Events
        fields = [
            'eventid', 'userid', 'type', 'title', 'category', 'starttime',
            'endtime', 'location', 'description', 'repeated', 'frequency',
            'startdate', 'enddate', 'emailto', 'emailcc', 'emailsubject',
            'emailbody', 'isactive', 'isdeleted', 'createdby', 'updatedby'
        ]
        
    def create(self, validated_data):
        validated_data['createdat'] = timezone.now()
        return super().create(validated_data)
    
    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        return super().update(instance, validated_data)

    def validate_emails(self, emails_str):
        """Helper method to validate multiple email addresses"""
        if not emails_str:
            return
        
        emails = [email.strip() for email in emails_str.split(',')]
        for email in emails:
            try:
                validate_email(email)
            except:
                raise serializers.ValidationError(
                    f"Invalid email format: {email}"
                )

    def validate(self, data):
        # Validate start date
        if not data.get('startdate'):
            raise serializers.ValidationError({
                "startdate": "Start date is required."
            })
        # Validate end date
        if not data.get('enddate'):
            raise serializers.ValidationError({
                "enddate": "End date is required."
            })
        # Validate date order
        if data.get('startdate') and data.get('enddate'):
            if data['startdate'] > data['enddate']:
                raise serializers.ValidationError({
                    "enddate": "End date must be after start date."
                })
            
        # Validate required fields for repeated events
        if data.get('repeated') == 1:
            # For repeated events, start and end dates cannot be the same
            if data['startdate'] == data['enddate']:
                raise serializers.ValidationError({
                    "enddate": "For repeated events, start and end dates cannot be the same."
                })
            
            # Validate frequency is provided for repeated events
            if not data.get('frequency'):
                raise serializers.ValidationError({
                    "frequency": "Frequency is required for repeated events."
                })

        # Validate required fields based on event type
        if data.get('type') == 'Action':
            # Validate email subject and body for Action type
            if not data.get('emailsubject'):
                raise serializers.ValidationError({
                    "emailsubject": "Email subject is required for Action type events."
                })
            if not data.get('emailbody'):
                raise serializers.ValidationError({
                    "emailbody": "Email body is required for Action type events."
                })


        # Validate times
        if data.get('starttime') and data.get('endtime'):
            if data['starttime'] > data['endtime']:
                raise serializers.ValidationError({
                    "endtime": "End time must be after start time."
                })

        # Validate email fields
        if data.get('emailto'):
            self.validate_emails(data['emailto'])
        if data.get('emailcc'):
            self.validate_emails(data['emailcc'])

        # Validate description length
        if data.get('description') and len(data['description']) > 2000:
            raise serializers.ValidationError({
                "description": "Description cannot exceed 2000 characters."
            })

        # Validate location length
        if data.get('location') and len(data['location']) > 255:
            raise serializers.ValidationError({
                "location": "Location cannot exceed 255 characters."
            })

        return data

class ExchangeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Exchange
        fields = [
            'exchangeid', 'name', 'display_name', 
            'isactive', 'isdeleted', 'createdby', 'updatedby', 'created_at', 'updated_at'
        ]
        read_only_fields = ['exchangeid', 'created_at', 'updated_at']

class TradingPairSerializer(serializers.ModelSerializer):
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    
    class Meta:
        model = TradingPair
        fields = [
            'tradingpairid', 'exchange', 'exchange_name', 'symbol', 
            'base_currency', 'quote_currency',
            'isactive', 'isdeleted', 'createdby', 'updatedby', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ['tradingpairid', 'created_at', 'updated_at']

class UserExchangeCredentialsSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    
    class Meta:
        model = UserExchangeCredentials
        fields = [
            'credentialid', 'user', 'username', 'exchange', 'exchange_name',
            'api_key', 'api_secret', 'passphrase', 'is_testnet',
            'isactive', 'isdeleted', 'createdby', 'updatedby',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['credentialid', 'created_at', 'updated_at']
        extra_kwargs = {
            'api_key': {'write_only': True},
            'api_secret': {'write_only': True},
            'passphrase': {'write_only': True},
        }

    def _get_encryption_key(self):
        """Get encryption key from settings or environment"""
        key = getattr(settings, 'CRYPTO_ENCRYPTION_KEY', os.getenv('CRYPTO_ENCRYPTION_KEY'))
        if not key:
            key = Fernet.generate_key()
        return key.encode() if isinstance(key, str) else key

    def _encrypt_credential(self, credential: str) -> str:
        """Encrypt API credential"""
        if not credential:
            return ""
        f = Fernet(self._get_encryption_key())
        return f.encrypt(credential.encode()).decode()

    def create(self, validated_data):
        """Encrypt credentials before saving"""
        validated_data['created_at'] = timezone.now()
        validated_data['updated_at'] = timezone.now()
        
        # Encrypt sensitive fields
        if 'api_key' in validated_data:
            validated_data['api_key'] = self._encrypt_credential(validated_data['api_key'])
        if 'api_secret' in validated_data:
            validated_data['api_secret'] = self._encrypt_credential(validated_data['api_secret'])
        if 'passphrase' in validated_data and validated_data['passphrase']:
            validated_data['passphrase'] = self._encrypt_credential(validated_data['passphrase'])
        
        # Set default values according to your model structure
        if 'isactive' not in validated_data:
            validated_data['isactive'] = 1
            
        return super().create(validated_data)

    def update(self, instance, validated_data):
        """Encrypt credentials when updating"""
        validated_data['updated_at'] = timezone.now()
        
        # Encrypt sensitive fields if they're being updated
        if 'api_key' in validated_data:
            validated_data['api_key'] = self._encrypt_credential(validated_data['api_key'])
        if 'api_secret' in validated_data:
            validated_data['api_secret'] = self._encrypt_credential(validated_data['api_secret'])
        if 'passphrase' in validated_data and validated_data['passphrase']:
            validated_data['passphrase'] = self._encrypt_credential(validated_data['passphrase'])
            
        return super().update(instance, validated_data)

class APIUsageSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    
    class Meta:
        model = APIUsage
        fields = [
            'apiusageid', 'user', 'username', 'exchange', 'exchange_name',
            'endpoint_type', 'request_count', 'timestamp',
            'isactive', 'isdeleted'
        ]
        read_only_fields = ['apiusageid', 'timestamp']

class UserTradeSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    profit_loss = serializers.SerializerMethodField()
    side_display = serializers.SerializerMethodField()
    trade_type_display = serializers.SerializerMethodField()
    status_display = serializers.SerializerMethodField()
    
    class Meta:
        model = UserTrade
        fields = [
            'tradeid', 'user', 'username', 'exchange', 'exchange_name',
            'symbol', 'side', 'side_display', 'trade_type', 'trade_type_display', 
            'amount', 'price', 'filled_amount', 'filled_price', 
            'status', 'status_display', 'exchange_order_id', 'fees', 'profit_loss',
            'isactive', 'isdeleted', 'createdby', 'updatedby',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['tradeid', 'created_at', 'updated_at']

    def get_side_display(self, obj):
        """Get display value for side"""
        return dict(obj._meta.get_field('side').choices).get(obj.side, obj.side)
    
    def get_trade_type_display(self, obj):
        """Get display value for trade_type"""
        return dict(obj._meta.get_field('trade_type').choices).get(obj.trade_type, obj.trade_type)
    
    def get_status_display(self, obj):
        """Get display value for status"""
        return dict(obj._meta.get_field('status').choices).get(obj.status, obj.status)

    def get_profit_loss(self, obj):
        """Calculate profit/loss for filled trades"""
        try:
            if obj.status == 'filled' and obj.filled_amount and obj.filled_price and obj.price:
                if obj.side == 'buy':
                    # For buy orders, loss if filled_price > price
                    return float((obj.price - obj.filled_price) * obj.filled_amount - obj.fees)
                else:  # sell
                    # For sell orders, profit if filled_price > price
                    return float((obj.filled_price - obj.price) * obj.filled_amount - obj.fees)
            return 0.0
        except (TypeError, ZeroDivisionError, AttributeError):
            return 0.0

class UserPortfolioSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    portfolio_value_usd = serializers.SerializerMethodField()
    unrealized_pnl = serializers.SerializerMethodField()
    
    class Meta:
        model = UserPortfolio
        fields = [
            'portfolioid', 'user', 'username', 'exchange', 'exchange_name',
            'currency', 'total_amount', 'available_amount', 'locked_amount',
            'avg_buy_price', 'portfolio_value_usd', 'unrealized_pnl',
            'isactive', 'isdeleted', 'createdby', 'updatedby',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['portfolioid', 'created_at', 'updated_at']

    def get_portfolio_value_usd(self, obj):
        """Calculate portfolio value in USD (placeholder)"""
        try:
            if obj.avg_buy_price and obj.total_amount:
                return float(obj.avg_buy_price * obj.total_amount)
            return 0.0
        except (TypeError, ZeroDivisionError, AttributeError):
            return 0.0

    def get_unrealized_pnl(self, obj):
        """Calculate unrealized P&L (placeholder - needs current market price)"""
        # In production, you would fetch current market price and calculate
        return 0.0

# User profile serializer for trading
class UserProfileSerializer(serializers.ModelSerializer):
    total_trades = serializers.SerializerMethodField()
    active_exchanges = serializers.SerializerMethodField()
    trading_experience_display = serializers.SerializerMethodField()
    risk_tolerance_display = serializers.SerializerMethodField()
    
    class Meta:
        model = Users
        fields = [
            'userid', 'fullname', 'email', 'trading_experience', 'trading_experience_display',
            'risk_tolerance', 'risk_tolerance_display', 'total_trades', 
            'active_exchanges', 'createdat', 'updatedat'
        ]
        read_only_fields = ['userid', 'createdat', 'updatedat']

    def get_trading_experience_display(self, obj):
        """Get display value for trading experience"""
        choices = [('beginner', 'Beginner'), ('intermediate', 'Intermediate'), ('advanced', 'Advanced')]
        return dict(choices).get(obj.trading_experience, obj.trading_experience)
    
    def get_risk_tolerance_display(self, obj):
        """Get display value for risk tolerance"""
        choices = [('low', 'Low'), ('medium', 'Medium'), ('high', 'High')]
        return dict(choices).get(obj.risk_tolerance, obj.risk_tolerance)
    def get_total_trades(self, obj):
        """Get total number of trades for this user"""
        return UserTrade.objects.filter(
            user=obj, 
            isactive=1, 
            isdeleted=0
        ).count()

    def get_active_exchanges(self, obj):
        """Get number of active exchanges for this user"""
        return UserExchangeCredentials.objects.filter(
            user=obj, 
            isactive=1, 
            isdeleted=0
        ).count()

# Dashboard summary serializer
class DashboardSerializer(serializers.Serializer):
    """Serializer for trading dashboard summary"""
    user_profile = UserProfileSerializer()
    subscription = SubscriptionSerializer()
    recent_trades = UserTradeSerializer(many=True)
    portfolio_summary = serializers.DictField()
    api_usage_stats = serializers.DictField()
    exchange_list = ExchangeSerializer(many=True)

# API Usage stats serializer
class APIUsageStatsSerializer(serializers.Serializer):
    """Serializer for API usage statistics"""
    used_calls = serializers.IntegerField()
    max_calls = serializers.IntegerField()
    remaining_calls = serializers.IntegerField()
    reset_time = serializers.DateTimeField()
    daily_usage = serializers.IntegerField(required=False)
    
    