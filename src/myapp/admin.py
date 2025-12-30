from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.db.models import Count, Sum
from .models import (
    Users, Subscriptionplans, Subscriptions, Renewals, Payments,
    Exchange, TradingPair, UserExchangeCredentials, UserTrade, UserPortfolio, APIUsage,
    Activitylogs, Auditlogs, Monthlyanalytics, Notifications,
    Reminders, Events
)

# =============================================================================
# USER MANAGEMENT
# =============================================================================

@admin.register(Users)
class UsersAdmin(admin.ModelAdmin):
    list_display = ('userid', 'fullname', 'email', 'role', 'trading_experience', 'is_active_status', 'createdat')
    list_filter = ('role', 'trading_experience', 'risk_tolerance', 'isactive', 'isdeleted')
    search_fields = ('fullname', 'email', 'organization')
    readonly_fields = ('userid', 'createdat', 'updatedat', 'passwordhash')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('fullname', 'email', 'role', 'passwordhash')
        }),
        ('Organization Details', {
            'fields': ('organization', 'phone', 'address', 'state', 'zipcode', 'country', 'logo_path'),
            'classes': ('collapse',)
        }),
        ('Trading Profile', {
            'fields': ('trading_experience', 'risk_tolerance'),
        }),
        ('SMTP Settings', {
            'fields': ('useusersmtp', 'smtphost', 'smtpport', 'smtphostuser', 'smtphostpassword', 'smtpusetls'),
            'classes': ('collapse',)
        }),
        ('System Fields', {
            'fields': ('isactive', 'isdeleted', 'createdby', 'updatedby', 'createdat', 'updatedat'),
            'classes': ('collapse',)
        }),
    )
    
    def is_active_status(self, obj):
        if obj.isactive == 1:
            return format_html('<span style="color: green;">Active</span>')
        return format_html('<span style="color: red;">Inactive</span>')
    is_active_status.short_description = 'Status'

# =============================================================================
# SUBSCRIPTION MANAGEMENT
# =============================================================================

@admin.register(Subscriptionplans)
class SubscriptionplansAdmin(admin.ModelAdmin):
    list_display = ('name', 'monthlyprice', 'yearlyprice', 'max_exchanges', 'ai_predictions_enabled', 'trade_automation', 'is_active_status')
    list_filter = ('ai_predictions_enabled', 'advanced_indicators_enabled', 'portfolio_tracking', 'trade_automation', 'isactive')
    search_fields = ('name', 'description')
    readonly_fields = ('subscriptionplanid', 'createdat', 'updatedat')
    
    fieldsets = (
        ('Plan Details', {
            'fields': ('name', 'description', 'monthlyprice', 'yearlyprice')
        }),
        ('Features & Limits', {
            'fields': ('max_exchanges', 'max_api_calls_per_hour', 'ai_predictions_enabled', 
                      'advanced_indicators_enabled', 'portfolio_tracking', 'trade_automation')
        }),
        ('Feature Details', {
            'fields': ('featuredetails',)
        }),
        ('System Fields', {
            'fields': ('isactive', 'isdeleted', 'createdby', 'updatedby', 'createdat', 'updatedat'),
            'classes': ('collapse',)
        }),
    )
    
    def is_active_status(self, obj):
        if obj.isactive == 1:
            return format_html('<span style="color: green;">Active</span>')
        return format_html('<span style="color: red;">Inactive</span>')
    is_active_status.short_description = 'Status'

@admin.register(Subscriptions)
class SubscriptionsAdmin(admin.ModelAdmin):
    list_display = ('userid', 'get_user_name', 'get_plan_name', 'status', 'billingfrequency', 'startdate', 'enddate', 'autorenew')
    list_filter = ('status', 'billingfrequency', 'autorenew', 'startdate', 'enddate')
    search_fields = ('userid__fullname', 'userid__email', 'subscriptionplanid__name')
    readonly_fields = ('subscriptionid', 'createdat', 'updatedat', 'lastrenewedat')
    raw_id_fields = ('userid', 'subscriptionplanid')
    
    def get_user_name(self, obj):
        return obj.userid.fullname if obj.userid else 'N/A'
    get_user_name.short_description = 'User Name'
    
    def get_plan_name(self, obj):
        return obj.subscriptionplanid.name if obj.subscriptionplanid else 'N/A'
    get_plan_name.short_description = 'Plan'

@admin.register(Renewals)
class RenewalsAdmin(admin.ModelAdmin):
    list_display = ('subscriptionid', 'get_user_name', 'renewaldate', 'renewalcost', 'get_renewed_by')
    list_filter = ('renewaldate', 'isactive')
    search_fields = ('subscriptionid__userid__fullname', 'renewedby__fullname')
    readonly_fields = ('renewalid', 'createdat', 'updatedat')
    raw_id_fields = ('subscriptionid', 'renewedby')
    
    def get_user_name(self, obj):
        return obj.subscriptionid.userid.fullname if obj.subscriptionid and obj.subscriptionid.userid else 'N/A'
    get_user_name.short_description = 'User'
    
    def get_renewed_by(self, obj):
        return obj.renewedby.fullname if obj.renewedby else 'System'
    get_renewed_by.short_description = 'Renewed By'

@admin.register(Payments)
class PaymentsAdmin(admin.ModelAdmin):
    list_display = ('paymentid', 'get_user_name', 'amount', 'paymentdate', 'paymentmethod', 'status')
    list_filter = ('status', 'paymentmethod', 'paymentdate')
    search_fields = ('subscriptionid__userid__fullname', 'referencenumber')
    readonly_fields = ('paymentid', 'createdat', 'updatedat')
    raw_id_fields = ('subscriptionid',)
    
    def get_user_name(self, obj):
        return obj.subscriptionid.userid.fullname if obj.subscriptionid and obj.subscriptionid.userid else 'N/A'
    get_user_name.short_description = 'User'

# =============================================================================
# TRADING MODELS
# =============================================================================

@admin.register(Exchange)
class ExchangeAdmin(admin.ModelAdmin):
    list_display = ('name', 'display_name', 'market_type', 'isactive', 'get_user_count')
    list_filter = ('isactive', 'isdeleted', 'market_type')
    search_fields = ('name', 'display_name')
    readonly_fields = ('exchangeid', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Exchange Information', {
            'fields': ('name', 'display_name', 'market_type')
        }),
        ('System Fields', {
            'fields': ('isactive', 'isdeleted', 'createdby', 'updatedby', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_user_count(self, obj):
        count = UserExchangeCredentials.objects.filter(exchange=obj, isactive=1).count()
        return f"{count} users"
    get_user_count.short_description = 'Connected Users'

@admin.register(TradingPair)
class TradingPairAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'get_exchange_name', 'base_currency', 'quote_currency', 'isactive')
    list_filter = ('exchange', 'isactive', 'base_currency', 'quote_currency')
    search_fields = ('symbol', 'base_currency', 'quote_currency', 'exchange__name')
    readonly_fields = ('tradingpairid', 'created_at', 'updated_at')
    raw_id_fields = ('exchange',)
    
    def get_exchange_name(self, obj):
        return obj.exchange.display_name
    get_exchange_name.short_description = 'Exchange'

@admin.register(UserExchangeCredentials)
class UserExchangeCredentialsAdmin(admin.ModelAdmin):
    list_display = ('get_user_name', 'get_exchange_name', 'is_testnet', 'isactive', 'created_at')
    list_filter = ('exchange', 'is_testnet', 'isactive', 'isactive')
    search_fields = ('user__fullname', 'user__email', 'exchange__name')
    readonly_fields = ('credentialid', 'created_at', 'updated_at')
    raw_id_fields = ('user', 'exchange')
    
    # Don't show sensitive fields in admin for security
    exclude = ('api_key', 'api_secret', 'passphrase')
    
    def get_user_name(self, obj):
        return obj.user.fullname if obj.user else 'N/A'
    get_user_name.short_description = 'User'
    
    def get_exchange_name(self, obj):
        return obj.exchange.display_name if obj.exchange else 'N/A'
    get_exchange_name.short_description = 'Exchange'

@admin.register(UserTrade)
class UserTradeAdmin(admin.ModelAdmin):
    list_display = ('get_user_name', 'get_exchange_name', 'symbol', 'side', 'amount', 'price', 'status', 'get_profit_loss', 'created_at')
    list_filter = ('exchange', 'side', 'trade_type', 'status', 'created_at')
    search_fields = ('user__fullname', 'user__email', 'symbol', 'exchange_order_id')
    readonly_fields = ('tradeid', 'created_at', 'updated_at')
    raw_id_fields = ('user', 'exchange')
    date_hierarchy = 'created_at'
    
    def get_user_name(self, obj):
        return obj.user.fullname if obj.user else 'N/A'
    get_user_name.short_description = 'User'
    
    def get_exchange_name(self, obj):
        return obj.exchange.display_name if obj.exchange else 'N/A'
    get_exchange_name.short_description = 'Exchange'
    
    def get_profit_loss(self, obj):
        if obj.status == 'filled' and obj.filled_amount and obj.filled_price and obj.price:
            try:
                if obj.side == 'buy':
                    pnl = float((obj.price - obj.filled_price) * obj.filled_amount - obj.fees)
                else:
                    pnl = float((obj.filled_price - obj.price) * obj.filled_amount - obj.fees)
                
                if pnl > 0:
                    return format_html('<span style="color: green;">+${:.2f}</span>', pnl)
                elif pnl < 0:
                    return format_html('<span style="color: red;">${:.2f}</span>', pnl)
                else:
                    return format_html('<span style="color: blue;">$0.00</span>')
            except:
                return 'N/A'
        return 'N/A'
    get_profit_loss.short_description = 'P&L'

@admin.register(UserPortfolio)
class UserPortfolioAdmin(admin.ModelAdmin):
    list_display = ('get_user_name', 'get_exchange_name', 'currency', 'total_amount', 'available_amount', 'locked_amount', 'avg_buy_price', 'get_portfolio_value')
    list_filter = ('exchange', 'currency', 'created_at')
    search_fields = ('user__fullname', 'user__email', 'currency', 'exchange__name')
    readonly_fields = ('portfolioid', 'created_at', 'updated_at')
    raw_id_fields = ('user', 'exchange')
    
    def get_user_name(self, obj):
        return obj.user.fullname if obj.user else 'N/A'
    get_user_name.short_description = 'User'
    
    def get_exchange_name(self, obj):
        return obj.exchange.display_name if obj.exchange else 'N/A'
    get_exchange_name.short_description = 'Exchange'
    
    def get_portfolio_value(self, obj):
        try:
            if obj.avg_buy_price and obj.total_amount:
                value = float(obj.avg_buy_price * obj.total_amount)
                return format_html('${:.2f}', value)
            return '$0.00'
        except:
            return 'N/A'
    get_portfolio_value.short_description = 'Portfolio Value'

@admin.register(APIUsage)
class APIUsageAdmin(admin.ModelAdmin):
    list_display = ('get_user_name', 'get_exchange_name', 'endpoint_type', 'request_count', 'timestamp')
    list_filter = ('exchange', 'endpoint_type', 'timestamp')
    search_fields = ('user__fullname', 'user__email', 'exchange__name')
    readonly_fields = ('apiusageid', 'timestamp')
    raw_id_fields = ('user', 'exchange')
    date_hierarchy = 'timestamp'
    
    def get_user_name(self, obj):
        return obj.user.fullname if obj.user else 'N/A'
    get_user_name.short_description = 'User'
    
    def get_exchange_name(self, obj):
        return obj.exchange.display_name if obj.exchange else 'N/A'
    get_exchange_name.short_description = 'Exchange'

# =============================================================================
# SYSTEM LOGS & ANALYTICS
# =============================================================================

@admin.register(Activitylogs)
class ActivitylogsAdmin(admin.ModelAdmin):
    list_display = ('activityid', 'get_user_name', 'activitytype', 'activitydate')
    list_filter = ('activitytype', 'activitydate', 'isactive')
    search_fields = ('userid__fullname', 'activitytype', 'activitydetails')
    readonly_fields = ('activityid', 'createdat', 'updatedat')
    raw_id_fields = ('userid',)
    date_hierarchy = 'activitydate'
    
    def get_user_name(self, obj):
        return obj.userid.fullname if obj.userid else 'System'
    get_user_name.short_description = 'User'

@admin.register(Auditlogs)
class AuditlogsAdmin(admin.ModelAdmin):
    list_display = ('auditlogid', 'get_user_name', 'action', 'tableaffected', 'recordid', 'createdat')
    list_filter = ('action', 'tableaffected', 'createdat')
    search_fields = ('userid__fullname', 'action', 'tableaffected')
    readonly_fields = ('auditlogid', 'createdat', 'updatedat')
    raw_id_fields = ('userid',)
    date_hierarchy = 'createdat'
    
    def get_user_name(self, obj):
        return obj.userid.fullname if obj.userid else 'System'
    get_user_name.short_description = 'User'

@admin.register(Monthlyanalytics)
class MonthlyanalyticsAdmin(admin.ModelAdmin):
    list_display = ('analyticsid', 'get_user_name', 'year', 'month', 'renewals', 'cancellations', 'newsubscriptions', 'totalpayments')
    list_filter = ('year', 'month', 'isactive')
    search_fields = ('userid__fullname',)
    readonly_fields = ('analyticsid', 'createdat', 'updatedat')
    raw_id_fields = ('userid',)
    
    def get_user_name(self, obj):
        return obj.userid.fullname if obj.userid else 'System'
    get_user_name.short_description = 'User'

@admin.register(Notifications)
class NotificationsAdmin(admin.ModelAdmin):
    list_display = ('notificationid', 'get_user_name', 'title', 'type', 'get_read_status', 'createdat')
    list_filter = ('type', 'isread', 'isactive', 'createdat')
    search_fields = ('userid__fullname', 'title', 'message')
    readonly_fields = ('notificationid', 'createdat', 'updatedat')
    raw_id_fields = ('userid',)
    
    def get_user_name(self, obj):
        return obj.userid.fullname if obj.userid else 'System'
    get_user_name.short_description = 'User'
    
    def get_read_status(self, obj):
        if obj.isread == 1:
            return format_html('<span style="color: green;">Read</span>')
        return format_html('<span style="color: orange;">● Unread</span>')
    get_read_status.short_description = 'Status'

@admin.register(Reminders)
class RemindersAdmin(admin.ModelAdmin):
    list_display = ('reminderid', 'get_user_name', 'note', 'timestamp', 'is_active_status')
    list_filter = ('timestamp', 'isactive')
    search_fields = ('userid__fullname', 'note')
    readonly_fields = ('reminderid', 'createdat', 'updatedat')
    raw_id_fields = ('userid',)
    date_hierarchy = 'timestamp'
    
    def get_user_name(self, obj):
        return obj.userid.fullname if obj.userid else 'System'
    get_user_name.short_description = 'User'
    
    def is_active_status(self, obj):
        if obj.isactive == 1:
            return format_html('<span style="color: green;">Active</span>')
        return format_html('<span style="color: red;">Inactive</span>')
    is_active_status.short_description = 'Status'

@admin.register(Events)
class EventsAdmin(admin.ModelAdmin):
    list_display = ('eventid', 'get_user_name', 'title', 'type', 'category', 'startdate', 'enddate', 'get_repeated_status')
    list_filter = ('type', 'category', 'repeated', 'frequency', 'startdate')
    search_fields = ('userid__fullname', 'title', 'description', 'location')
    readonly_fields = ('eventid', 'createdat', 'updatedat')
    raw_id_fields = ('userid',)
    date_hierarchy = 'startdate'
    
    def get_user_name(self, obj):
        return obj.userid.fullname if obj.userid else 'System'
    get_user_name.short_description = 'User'
    
    def get_repeated_status(self, obj):
        if obj.repeated == 1:
            return format_html('<span style="color: blue;">{}</span>', obj.frequency or 'N/A')
        return format_html('<span style="color: gray;">No</span>')
    get_repeated_status.short_description = 'Repeated'

# Admin site customization
admin.site.site_header = 'Crypto Trading Platform Admin'
admin.site.site_title = 'Crypto Trading Admin'
admin.site.index_title = 'Welcome to Crypto Trading Platform Administration'