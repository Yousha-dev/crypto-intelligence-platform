from decimal import Decimal
from rest_framework import serializers
from myapp.models import (
    Subscriptionplans, SUBSCRIPTION_STATUS_CHOICES, BILLING_FREQUENCY_CHOICES, 
    Subscriptions, Renewals, Payments, Users, Exchange, 
    UserExchangeCredentials, APIUsage, UserTrade, 
    UserPortfolio, TradingPair, PAYMENT_STATUS_CHOICES, PAYMENT_METHOD_CHOICES,
    TRADE_SIDES, TRADE_TYPES, TRADE_STATUS
)
from django.utils import timezone
from django.core.exceptions import ValidationError

class SubscriptionPlanSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    
    class Meta:
        model = Subscriptionplans
        fields = [
            'subscriptionplanid', 'name', 'description', 'monthlyprice', 'yearlyprice',
            'max_exchanges', 'max_api_calls_per_hour', 'ai_predictions_enabled',
            'advanced_indicators_enabled', 'portfolio_tracking', 'trade_automation',
            'featuredetails', 'isactive', 'isdeleted', 'createdby', 'updatedby', 
            'createdat', 'updatedat'
        ]
        read_only_fields = ('subscriptionplanid', 'createdat', 'updatedat')

    def create(self, validated_data):
        validated_data['createdat'] = timezone.now()
        validated_data['updatedat'] = timezone.now()
        return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        return super().update(instance, validated_data)
    
    def validate(self, data):
        """Validate subscription plan data"""
        # Validate prices
        if 'monthlyprice' in data and data['monthlyprice'] < 0:
            raise serializers.ValidationError({
                "monthlyprice": "Monthly price cannot be negative."
            })
        
        if 'yearlyprice' in data and data['yearlyprice'] < 0:
            raise serializers.ValidationError({
                "yearlyprice": "Yearly price cannot be negative."
            })

        # Validate trading specific limits
        if 'max_exchanges' in data and data['max_exchanges'] <= 0:
            raise serializers.ValidationError({
                "max_exchanges": "Maximum exchanges must be greater than 0."
            })

        if 'max_api_calls_per_hour' in data and data['max_api_calls_per_hour'] <= 0:
            raise serializers.ValidationError({
                "max_api_calls_per_hour": "Maximum API calls per hour must be greater than 0."
            })

        # Validate name
        if 'name' in data and not data['name'].strip():
            raise serializers.ValidationError({
                "name": "Name cannot be empty."
            })

        # Validate feature combinations
        if data.get('trade_automation') and not data.get('ai_predictions_enabled'):
            raise serializers.ValidationError({
                "trade_automation": "Trade automation requires AI predictions to be enabled."
            })

        if data.get('advanced_indicators_enabled') and not data.get('ai_predictions_enabled'):
            raise serializers.ValidationError({
                "advanced_indicators_enabled": "Advanced indicators require AI predictions to be enabled."
            })

        return data

class SubscriptionSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    status = serializers.ChoiceField(choices=SUBSCRIPTION_STATUS_CHOICES)
    billingfrequency = serializers.ChoiceField(choices=BILLING_FREQUENCY_CHOICES)
    username = serializers.CharField(source='userid.fullname', read_only=True)
    user_email = serializers.CharField(source='userid.email', read_only=True)
    plan_name = serializers.CharField(source='subscriptionplanid.name', read_only=True)
    plan_price = serializers.DecimalField(source='subscriptionplanid.monthlyprice', max_digits=10, decimal_places=2, read_only=True)

    class Meta:
        model = Subscriptions
        fields = [
            'subscriptionid', 'userid', 'username', 'user_email', 'subscriptionplanid', 'plan_name', 'plan_price',
            'billingfrequency', 'startdate', 'enddate', 'autorenew', 'status', 
            'renewalcount', 'lastrenewedat', 'isactive', 'isdeleted', 
            'createdby', 'updatedby', 'createdat', 'updatedat'
        ]
        read_only_fields = ('subscriptionid', 'createdat', 'updatedat', 'lastrenewedat')

    def create(self, validated_data):
        validated_data['createdat'] = timezone.now()
        validated_data['updatedat'] = timezone.now()
        return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        return super().update(instance, validated_data)
    
    def validate(self, data):
        """Validate subscription data"""
        # Validate dates
        if 'startdate' in data and 'enddate' in data:
            if data['enddate'] <= data['startdate']:
                raise serializers.ValidationError({
                    "enddate": "End date must be after start date."
                })

        # Validate renewal count
        if 'renewalcount' in data and data['renewalcount'] < 0:
            raise serializers.ValidationError({
                "renewalcount": "Renewal count cannot be negative."
            })

        if not self.instance:  # create operation
            # For new subscriptions, require these fields
            if not data.get('subscriptionplanid'):
                raise serializers.ValidationError({
                    "subscriptionplanid": "Subscription plan ID is required for new subscriptions."
                })
            if not data.get('userid'):
                raise serializers.ValidationError({
                    "userid": "User ID is required for new subscriptions."
                })

        # Validate user doesn't have multiple active subscriptions
        if 'userid' in data and data.get('status') == 'Active':
            existing_subscriptions = Subscriptions.objects.filter(
                userid=data['userid'],
                status='Active',
                isactive=1,
                isdeleted=0
            )
            if self.instance:
                existing_subscriptions = existing_subscriptions.exclude(
                    subscriptionid=self.instance.subscriptionid
                )
            
            if existing_subscriptions.exists():
                raise serializers.ValidationError({
                    "userid": "User already has an active subscription."
                })

        return data

class RenewalSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    subscription_info = serializers.CharField(source='subscriptionid.userid.fullname', read_only=True)
    user_email = serializers.CharField(source='subscriptionid.userid.email', read_only=True)
    renewed_by_name = serializers.CharField(source='renewedby.fullname', read_only=True)
    plan_name = serializers.CharField(source='subscriptionid.subscriptionplanid.name', read_only=True)
    
    class Meta:
        model = Renewals
        fields = [
            'renewalid', 'subscriptionid', 'subscription_info', 'user_email', 'plan_name',
            'renewedby', 'renewed_by_name', 'renewaldate', 'renewalcost', 'notes', 
            'isactive', 'isdeleted', 'createdat', 'updatedat'
        ]
        read_only_fields = ('renewalid', 'createdat', 'updatedat')

    def create(self, validated_data):
        validated_data['createdat'] = timezone.now()
        validated_data['updatedat'] = timezone.now()
        return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        return super().update(instance, validated_data)
    
    def validate(self, data):
        """Validate renewal data"""
        # Validate renewal cost
        if 'renewalcost' in data and data['renewalcost'] < Decimal('0.00'):
            raise serializers.ValidationError({
                "renewalcost": "Renewal cost cannot be negative."
            })

        # Validate renewal date
        if 'renewaldate' in data:
            if data['renewaldate'] > timezone.now():
                raise serializers.ValidationError({
                    "renewaldate": "Renewal date cannot be in the future."
                })

        # Validate required relationships
        if not self.instance:  # create operation
            if not data.get('subscriptionid'):
                raise serializers.ValidationError({
                    "subscriptionid": "Subscription ID is required."
                })
        
        if 'subscriptionid' in data:
            subscription = data['subscriptionid']
            if hasattr(subscription, 'isdeleted') and subscription.isdeleted == 1:
                raise serializers.ValidationError({
                    "subscriptionid": "Cannot renew a deleted subscription."
                })
            if hasattr(subscription, 'isactive') and subscription.isactive == 0:
                raise serializers.ValidationError({
                    "subscriptionid": "Cannot renew an inactive subscription."
                })

        return data
        
class PaymentSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    subscription_info = serializers.CharField(source='subscriptionid.userid.fullname', read_only=True)
    user_email = serializers.CharField(source='subscriptionid.userid.email', read_only=True)
    plan_name = serializers.CharField(source='subscriptionid.subscriptionplanid.name', read_only=True)
    paymentmethod = serializers.ChoiceField(choices=PAYMENT_METHOD_CHOICES, required=False)
    status = serializers.ChoiceField(choices=PAYMENT_STATUS_CHOICES, required=False)
    
    class Meta:
        model = Payments
        fields = [
            'paymentid', 'subscriptionid', 'subscription_info', 'user_email', 'plan_name',
            'amount', 'paymentdate', 'paymentmethod', 'referencenumber', 'status', 
            'paymentresponse', 'isactive', 'isdeleted', 'createdby', 'updatedby', 
            'createdat', 'updatedat'
        ]
        read_only_fields = ('paymentid', 'createdat', 'updatedat')
    
    def create(self, validated_data):
        validated_data['createdat'] = timezone.now()
        validated_data['updatedat'] = timezone.now()
        return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        return super().update(instance, validated_data)
        
    def validate(self, data):
        """Validate payment data"""
        # Validate amount
        if 'amount' in data and data['amount'] <= Decimal('0.00'):
            raise serializers.ValidationError({
                "amount": "Payment amount must be greater than zero."
            })

        # Validate payment date
        if 'paymentdate' in data:
            if data['paymentdate'] > timezone.now().date():
                raise serializers.ValidationError({
                    "paymentdate": "Payment date cannot be in the future."
                })

        # Validate required fields
        if not self.instance:  # create operation
            if not data.get('subscriptionid'):
                raise serializers.ValidationError({
                    "subscriptionid": "Subscription ID is required."
                })

        return data

# Trading Admin Serializers
class ExchangeAdminSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    user_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Exchange
        fields = [
            'exchangeid', 'name', 'display_name', 
            'user_count', 'isactive', 'isdeleted', 'createdby', 'updatedby', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ('exchangeid', 'created_at', 'updated_at')

    def get_user_count(self, obj):
        return UserExchangeCredentials.objects.filter(
            exchange=obj, 
            isactive=1, 
            isdeleted=0
        ).count()

    def validate(self, data):
        """Validate exchange data"""
        if 'name' in data and not data['name'].strip():
            raise serializers.ValidationError({
                "name": "Exchange name cannot be empty."
            })
            
        return data


class UserExchangeCredentialsAdminSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    user_email = serializers.CharField(source='user.email', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    
    class Meta:
        model = UserExchangeCredentials
        fields = [
            'credentialid', 'user', 'username', 'user_email', 'exchange', 'exchange_name',
            'is_testnet', 'isactive', 'isdeleted',
            'createdby', 'updatedby', 'created_at', 'updated_at'
        ]
        read_only_fields = ('credentialid', 'created_at', 'updated_at')
        # Don't expose encrypted credentials in admin
        
    def validate(self, data):
        """Validate user exchange credentials"""
        # Check for duplicate user-exchange combinations
        if 'user' in data and 'exchange' in data:
            existing = UserExchangeCredentials.objects.filter(
                user=data['user'],
                exchange=data['exchange'],
                isactive=1,
                isdeleted=0
            )
            if self.instance:
                existing = existing.exclude(credentialid=self.instance.credentialid)
            
            if existing.exists():
                raise serializers.ValidationError({
                    "exchange": "User already has credentials for this exchange."
                })
        
        return data

class APIUsageAdminSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    user_email = serializers.CharField(source='user.email', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    
    class Meta:
        model = APIUsage
        fields = [
            'apiusageid', 'user', 'username', 'user_email', 'exchange', 'exchange_name',
            'endpoint_type', 'request_count', 'timestamp', 'isactive', 'isdeleted'
        ]
        read_only_fields = ('apiusageid', 'timestamp')

class UserTradeAdminSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    user_email = serializers.CharField(source='user.email', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    profit_loss = serializers.SerializerMethodField()
    side = serializers.ChoiceField(choices=TRADE_SIDES)
    trade_type = serializers.ChoiceField(choices=TRADE_TYPES)
    status = serializers.ChoiceField(choices=TRADE_STATUS)
    
    class Meta:
        model = UserTrade
        fields = [
            'tradeid', 'user', 'username', 'user_email', 'exchange', 'exchange_name',
            'symbol', 'side', 'trade_type', 'amount', 'price', 'filled_amount', 
            'filled_price', 'status', 'exchange_order_id', 'fees', 'profit_loss', 
            'isactive', 'isdeleted', 'createdby', 'updatedby', 'created_at', 'updated_at'
        ]
        read_only_fields = ('tradeid', 'created_at', 'updated_at')

    def get_profit_loss(self, obj):
        """Calculate profit/loss for filled trades"""
        try:
            if obj.status == 'filled' and obj.filled_amount and obj.filled_price and obj.price:
                if obj.side == 'buy':
                    return float((obj.price - obj.filled_price) * obj.filled_amount - obj.fees)
                else:  # sell
                    return float((obj.filled_price - obj.price) * obj.filled_amount - obj.fees)
            return 0.0
        except (TypeError, ZeroDivisionError, AttributeError):
            return 0.0

class UserPortfolioAdminSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.fullname', read_only=True)
    user_email = serializers.CharField(source='user.email', read_only=True)
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    portfolio_value = serializers.SerializerMethodField()
    profit_loss_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = UserPortfolio
        fields = [
            'portfolioid', 'user', 'username', 'user_email', 'exchange', 'exchange_name',
            'currency', 'total_amount', 'available_amount', 'locked_amount', 'avg_buy_price', 
            'portfolio_value', 'profit_loss_percentage', 'isactive', 'isdeleted',
            'createdby', 'updatedby', 'created_at', 'updated_at'
        ]
        read_only_fields = ('portfolioid', 'created_at', 'updated_at')

    def get_portfolio_value(self, obj):
        """Calculate portfolio value"""
        try:
            if obj.avg_buy_price and obj.total_amount:
                return float(obj.avg_buy_price * obj.total_amount)
            return 0.0
        except (TypeError, ZeroDivisionError, AttributeError):
            return 0.0

    def get_profit_loss_percentage(self, obj):
        """Calculate profit/loss percentage (placeholder - would need current price)"""
        # This would need current market price to calculate actual P&L
        return 0.0

class TradingPairAdminSerializer(serializers.ModelSerializer):
    exchange_name = serializers.CharField(source='exchange.display_name', read_only=True)
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    trade_count = serializers.SerializerMethodField()
    
    class Meta:
        model = TradingPair
        fields = [
            'tradingpairid', 'exchange', 'exchange_name', 'symbol', 'base_currency', 
            'quote_currency', 'trade_count',
            'isactive', 'isdeleted', 'createdby', 'updatedby', 'created_at', 'updated_at'
        ]
        read_only_fields = ('tradingpairid', 'created_at', 'updated_at')

    def get_trade_count(self, obj):
        """Get number of trades for this pair"""
        return UserTrade.objects.filter(
            symbol=obj.symbol, 
            exchange=obj.exchange,
            isactive=1,
            isdeleted=0
        ).count()

    def validate(self, data):
        """Validate trading pair data"""
        if 'symbol' in data and not data['symbol'].strip():
            raise serializers.ValidationError({
                "symbol": "Symbol cannot be empty."
            })
            
        # Check for duplicate symbol-exchange combinations
        if 'symbol' in data and 'exchange' in data:
            existing = TradingPair.objects.filter(
                symbol=data['symbol'],
                exchange=data['exchange'],
                isactive=1,
                isdeleted=0
            )
            if self.instance:
                existing = existing.exclude(tradingpairid=self.instance.tradingpairid)
            
            if existing.exists():
                raise serializers.ValidationError({
                    "symbol": "This trading pair already exists for this exchange."
                })
        
        return data

# Dashboard/Statistics Serializers for Admin
class AdminDashboardStatsSerializer(serializers.Serializer):
    """Admin dashboard statistics"""
    total_users = serializers.IntegerField()
    active_subscriptions = serializers.IntegerField()
    total_trades = serializers.IntegerField()
    total_api_calls = serializers.IntegerField()
    revenue_this_month = serializers.DecimalField(max_digits=10, decimal_places=2)
    top_exchanges = serializers.ListField(child=serializers.DictField())
    subscription_distribution = serializers.DictField()
    recent_activities = serializers.ListField(child=serializers.DictField())
    active_models = serializers.IntegerField()
    total_portfolio_value = serializers.DecimalField(max_digits=20, decimal_places=2)

class UserDetailAdminSerializer(serializers.ModelSerializer):
    """Detailed user serializer for admin"""
    logo_url = serializers.SerializerMethodField()
    subscription_status = serializers.SerializerMethodField()
    total_trades = serializers.SerializerMethodField()
    portfolio_value = serializers.SerializerMethodField()
    
    class Meta:
        model = Users
        fields = [
            'userid', 'fullname', 'email', 'role', 'organization', 'phone', 'address',
            'state', 'zipcode', 'country', 'logo_url', 'trading_experience', 
            'risk_tolerance', 'subscription_status', 'total_trades', 'portfolio_value',
            'isactive', 'isdeleted', 'createdat', 'updatedat'
        ]

    def get_logo_url(self, obj):
        if obj.logo_path:
            from django.conf import settings
            return f"{settings.MEDIA_URL}{obj.logo_path}"
        return None

    def get_subscription_status(self, obj):
        try:
            subscription = Subscriptions.objects.filter(
                userid=obj.userid, 
                isactive=1, 
                isdeleted=0
            ).first()
            return subscription.status if subscription else 'No Subscription'
        except:
            return 'No Subscription'

    def get_total_trades(self, obj):
        return UserTrade.objects.filter(user=obj, isactive=1, isdeleted=0).count()

    def get_portfolio_value(self, obj):
        try:
            portfolios = UserPortfolio.objects.filter(user=obj, isactive=1, isdeleted=0)
            total_value = 0
            for portfolio in portfolios:
                if portfolio.avg_buy_price and portfolio.total_amount:
                    total_value += float(portfolio.avg_buy_price * portfolio.total_amount)
            return total_value
        except:
            return 0.0