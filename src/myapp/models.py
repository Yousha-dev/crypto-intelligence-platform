
# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models
from django.forms import ValidationError

 
BILLING_FREQUENCY_CHOICES = [
    ('Monthly', 'Monthly'),
    ('Yearly', 'Yearly'),
    ('Weekly', 'Weekly'),
    ('Semi-Annually', 'Semi-Annually'),
    ('Quarterly', 'Quarterly'),
    ('One-Time', 'One-Time'),
    ('Other', 'Other'),
]

SUBSCRIPTION_STATUS_CHOICES = [
    ('Active', 'Active'),
    ('Expired', 'Expired'),
    ('Cancelled', 'Cancelled'),
    ('Pending', 'Pending'),
    ('Suspended', 'Suspended'),
    ('RenewalPending', 'RenewalPending'),
    ('Trial', 'Trial'),
]

PAYMENT_STATUS_CHOICES = [
    ('Pending', 'Pending'),
    ('Completed', 'Completed'),
    ('Failed', 'Failed'),
]

PAYMENT_METHOD_CHOICES = [
    ('CreditCard', 'CreditCard'),
    ('PayPal', 'PayPal'),
    ('BankTransfer', 'BankTransfer'),
]

NOTIFICATIONS_TYPE_CHOICES = [
    ('Expiry', 'Expiry'),
    ('Renewal', 'Renewal'),
    ('System', 'System')
]

EVENTS_TYPE_CHOICES = [
    ('Action', 'Action'),
    ('Reminder', 'Reminder')
]

EVENTS_CATEGORY_CHOICES = [
    ('Personal', 'Personal'),
    ('Work', 'Work'),
    ('Birthday', 'Birthday'),
    ('Deadline', 'Deadline'),
    ('Other', 'Other')
]

EVENTS_FREQUENCY_CHOICES = [
    ('Daily', 'Daily'),
    ('Weekly', 'Weekly'),
    ('Monthly', 'Monthly'),
    ('Yearly', 'Yearly')
]

TRADE_SIDES = [
    ('buy', 'Buy'),
    ('sell', 'Sell'),
]

TRADE_TYPES = [
    ('market', 'Market'),
    ('limit', 'Limit'),
    ('stop', 'Stop'),
]

TRADE_STATUS = [
    ('pending', 'Pending'),
    ('filled', 'Filled'),
    ('cancelled', 'Cancelled'),
    ('failed', 'Failed'),
]


class Activitylogs(models.Model):
    activityid = models.AutoField(db_column='ActivityID', primary_key=True)  # Field name made lowercase.
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='UserID', blank=True, null=True)  # Field name made lowercase.
    activitytype = models.CharField(db_column='ActivityType', max_length=18)  # Field name made lowercase.
    activitydetails = models.TextField(db_column='ActivityDetails', blank=True, null=True)  # Field name made lowercase.
    activitydate = models.DateTimeField(db_column='ActivityDate', blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'ActivityLogs'
        indexes = [
            models.Index(fields=['userid', 'activitydate']),
            models.Index(fields=['activitytype', 'activitydate']),
        ]


class Auditlogs(models.Model):
    auditlogid = models.AutoField(db_column='AuditLogID', primary_key=True)  # Field name made lowercase.
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='UserID', blank=True, null=True)  # Field name made lowercase.
    action = models.CharField(db_column='Action', max_length=255)  # Field name made lowercase.
    tableaffected = models.CharField(db_column='TableAffected', max_length=255, blank=True, null=True)  # Field name made lowercase.
    recordid = models.IntegerField(db_column='RecordID', blank=True, null=True)  # Field name made lowercase.
    createdat = models.DateTimeField(db_column='CreatedAt')  # Field name made lowercase.
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    updatedat = models.DateTimeField(db_column='UpdatedAt')  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'AuditLogs'
        indexes = [
            models.Index(fields=['userid', 'createdat']),
            models.Index(fields=['tableaffected', 'createdat']),
        ]


class Monthlyanalytics(models.Model):
    analyticsid = models.AutoField(db_column='AnalyticsID', primary_key=True)  # Field name made lowercase.
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='UserID', blank=True, null=True)
    year = models.IntegerField(db_column='Year')  # Field name made lowercase.
    month = models.IntegerField(db_column='Month')  # Field name made lowercase.
    renewals = models.IntegerField(db_column='Renewals', blank=True, null=True)  # Field name made lowercase.
    cancellations = models.IntegerField(db_column='Cancellations', blank=True, null=True)  # Field name made lowercase.
    newsubscriptions = models.IntegerField(db_column='NewSubscriptions', blank=True, null=True)  # Field name made lowercase.
    totalpayments = models.DecimalField(db_column='TotalPayments', max_digits=10, decimal_places=2, blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'MonthlyAnalytics'

    
class Notifications(models.Model):
    notificationid = models.AutoField(db_column='NotificationID', primary_key=True)  # Field name made lowercase.
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='UserID', blank=True, null=True)
    title = models.TextField(db_column='Title')  # Field name made lowercase.
    message = models.TextField(db_column='Message')  # Field name made lowercase.
    type = models.CharField(
        db_column='Type',
        max_length=7,
        choices=NOTIFICATIONS_TYPE_CHOICES
    ) 
    isread = models.IntegerField(db_column='IsRead', blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Notifications'
        indexes = [
            models.Index(fields=['userid', 'isread', 'createdat']),
            models.Index(fields=['type', 'createdat']),
        ]
        
    def clean(self):
        if self.type not in dict(NOTIFICATIONS_TYPE_CHOICES):
            raise ValidationError({
                'type': 'Invalid notification type selected.'
            })

class Payments(models.Model):
    paymentid = models.AutoField(db_column='PaymentID', primary_key=True)  # Field name made lowercase. Field name made lowercase.
    subscriptionid = models.ForeignKey('Subscriptions', models.DO_NOTHING, db_column='SubscriptionID', blank=True, null=True)  # Field name made lowercase.
    amount = models.DecimalField(db_column='Amount', max_digits=10, decimal_places=2)  # Field name made lowercase.
    paymentdate = models.DateField(db_column='PaymentDate')  # Field name made lowercase.
    paymentmethod = models.CharField(
        db_column='PaymentMethod',
        max_length=12,
        choices=PAYMENT_METHOD_CHOICES,
        blank=True,
        null=True
    )  # Field name made lowercase.
    referencenumber = models.CharField(db_column='ReferenceNumber', max_length=255, blank=True, null=True)  # Field name made lowercase.
    status = models.CharField(
        db_column='Status',
        max_length=9,
        choices=PAYMENT_STATUS_CHOICES,
        blank=True,
        null=True
    )  # Field name made lowercase.
    paymentresponse = models.TextField(db_column='PaymentResponse', blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Payments'
        indexes = [
            models.Index(fields=['subscriptionid', 'paymentdate']),
            models.Index(fields=['status', 'paymentdate']),
        ]
        
    def clean(self):
        if self.paymentmethod and self.paymentmethod not in dict(PAYMENT_METHOD_CHOICES):
            raise ValidationError({
                'paymentmethod': 'Invalid payment method selected.'
            })
        if self.status and self.status not in dict(PAYMENT_STATUS_CHOICES):
            raise ValidationError({
                'status': 'Invalid payment status selected.'
            })

class Reminders(models.Model):
    reminderid = models.AutoField(db_column='ReminderID', primary_key=True)  # Field name made lowercase.
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='UserID', blank=True, null=True)  # Field name made lowercase.
    note = models.TextField(db_column='Note', blank=True, null=True)  # Field name made lowercase.
    timestamp = models.DateTimeField(db_column='Timestamp')  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Reminders'


class Events(models.Model):
    eventid = models.AutoField(db_column='EventID', primary_key=True)  # Field name made lowercase.
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='UserID', blank=True, null=True)
    type = models.CharField(db_column='Type', max_length=8, choices=EVENTS_TYPE_CHOICES)  # Field name made lowercase.
    title = models.TextField(db_column='Title')  # Field name made lowercase.
    category = models.CharField(db_column='Category', max_length=8, choices=EVENTS_CATEGORY_CHOICES)  # Field name made lowercase.
    starttime = models.TimeField(db_column='StartTime')  # Field name made lowercase.
    endtime = models.TimeField(db_column='EndTime')  # Field name made lowercase.
    location = models.CharField(db_column='Location', max_length=255, blank=True, null=True)  # Field name made lowercase.
    description = models.TextField(db_column='Description', blank=True, null=True)  # Field name made lowercase.
    repeated = models.IntegerField(db_column='Repeated')  # Field name made lowercase.
    frequency = models.CharField(db_column='Frequency', max_length=7, choices=EVENTS_FREQUENCY_CHOICES, blank=True, null=True)  # Field name made lowercase.
    startdate = models.DateField(db_column='StartDate')  # Field name made lowercase.
    enddate = models.DateField(db_column='EndDate', blank=True, null=True)  # Field name made lowercase.
    emailto = models.TextField(db_column='EmailTo')  # Field name made lowercase.
    emailcc = models.TextField(db_column='EmailCC', blank=True, null=True)  # Field name made lowercase.
    emailsubject = models.TextField(db_column='EmailSubject', blank=True, null=True)  # Field name made lowercase.
    emailbody = models.TextField(db_column='EmailBody', blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.


    class Meta:
        managed = True
        db_table = 'Events'
        
    def clean(self):
        if self.type not in dict(EVENTS_TYPE_CHOICES):
            raise ValidationError({
                'type': 'Invalid event type selected.'
            })
        if self.category not in dict(EVENTS_CATEGORY_CHOICES):
            raise ValidationError({
                'category': 'Invalid event category selected.'
            })
        if self.frequency and self.frequency not in dict(EVENTS_FREQUENCY_CHOICES):
            raise ValidationError({
                'frequency': 'Invalid event frequency selected.'
            })
        
        
class Renewals(models.Model):
    renewalid = models.AutoField(db_column='RenewalID', primary_key=True)  # Field name made lowercase.
    subscriptionid = models.ForeignKey('Subscriptions', models.DO_NOTHING, db_column='SubscriptionID', blank=True, null=True)  # Field name made lowercase.
    renewedby = models.ForeignKey('Users', models.DO_NOTHING, db_column='RenewedBy', blank=True, null=True)  # Field name made lowercase.
    renewaldate = models.DateTimeField(db_column='RenewalDate', blank=True, null=True)  # Field name made lowercase.
    renewalcost = models.DecimalField(db_column='RenewalCost', max_digits=10, decimal_places=2)  # Field name made lowercase.
    notes = models.TextField(db_column='Notes', blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(db_column='IsActive', blank=True, null=True)  # Field name made lowercase.
    isdeleted = models.IntegerField(db_column='IsDeleted', blank=True, null=True)  # Field name made lowercase.
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Renewals'



class Subscriptionplans(models.Model):
    subscriptionplanid = models.AutoField(db_column='SubscriptionPlanID', primary_key=True)
    name = models.CharField(db_column='Name', max_length=255)
    description = models.TextField(db_column='Description', blank=True, null=True)
    monthlyprice = models.DecimalField(db_column='MonthlyPrice', max_digits=10, decimal_places=2)
    yearlyprice = models.DecimalField(db_column='YearlyPrice', max_digits=10, decimal_places=2)
    max_exchanges = models.IntegerField(db_column='MaxExchanges', default=1)
    max_api_calls_per_hour = models.IntegerField(db_column='MaxAPICallsPerHour', default=100)
    ai_predictions_enabled = models.BooleanField(db_column='AIPredictionsEnabled', default=False)
    advanced_indicators_enabled = models.BooleanField(db_column='AdvancedIndicatorsEnabled', default=False)
    portfolio_tracking = models.BooleanField(db_column='PortfolioTracking', default=False)
    trade_automation = models.BooleanField(db_column='TradeAutomation', default=False)
    featuredetails = models.TextField(db_column='FeatureDetails')
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'SubscriptionPlans'


class Subscriptions(models.Model):
    subscriptionid = models.AutoField(db_column='SubscriptionID', primary_key=True)  # Field name made lowercase.
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='UserID', blank=True, null=True)
    subscriptionplanid = models.ForeignKey(Subscriptionplans, models.DO_NOTHING, db_column='SubscriptionPlanID', blank=True, null=True)  # Field name made lowercase.
    billingfrequency = models.CharField(
        db_column='BillingFrequency',
        max_length=13,
        choices=BILLING_FREQUENCY_CHOICES
    )  # Field name made lowercase.
    startdate = models.DateField(db_column='StartDate')  # Field name made lowercase.
    enddate = models.DateField(db_column='EndDate')  # Field name made lowercase.
    autorenew = models.IntegerField(db_column='AutoRenew')  # Field name made lowercase.
    status = models.CharField(
        db_column='Status',
        max_length=14,
        choices=SUBSCRIPTION_STATUS_CHOICES
    )  # Field name made lowercase.
    renewalcount = models.IntegerField(db_column='RenewalCount', blank=True, null=True)  # Field name made lowercase.
    lastrenewedat = models.DateTimeField(db_column='LastRenewedAt', blank=True, null=True)  # Field name made lowercase.
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)  # Field name made lowercase.
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)  # Field name made lowercase.
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)  # Field name made lowercase.
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Subscriptions'
        indexes = [
            models.Index(fields=['userid', 'status']),
            models.Index(fields=['enddate', 'status']),
            models.Index(fields=['status', 'autorenew']),
        ]

    def clean(self):
        if self.billingfrequency not in dict(BILLING_FREQUENCY_CHOICES):
            raise ValidationError({
                'billingfrequency': 'Invalid billing frequency selected.'
            })
        elif self.status not in dict(SUBSCRIPTION_STATUS_CHOICES):
            raise ValidationError({
                'status': 'Invalid subscription status selected.'
            })
 
class Users(models.Model):
    userid = models.AutoField(db_column='UserID', primary_key=True)
    fullname = models.CharField(db_column='FullName', max_length=255)
    email = models.CharField(db_column='Email', unique=True, max_length=255)
    passwordhash = models.CharField(db_column='PasswordHash', max_length=255)
    role = models.CharField(db_column='Role', max_length=12)
    
    # User organization fields
    organization = models.CharField(db_column='Organization', max_length=255, blank=True, null=True)
    phone = models.CharField(db_column='Phone', max_length=20, blank=True, null=True)
    address = models.TextField(db_column='Address', blank=True, null=True)
    state = models.CharField(db_column='State', max_length=45, blank=True, null=True)
    zipcode = models.CharField(db_column='ZipCode', max_length=45, blank=True, null=True)
    country = models.CharField(db_column='Country', max_length=45, blank=True, null=True)
    logo_path = models.CharField(db_column='LogoPath', max_length=500, blank=True, null=True)
    
    # SMTP settings
    useusersmtp = models.IntegerField(db_column='UseCustomSMTP', blank=True, null=True)
    smtphost = models.CharField(db_column='SMTPHost', max_length=255, blank=True, null=True)
    smtpport = models.IntegerField(db_column='SMTPPort', blank=True, null=True)
    smtphostuser = models.CharField(db_column='SMTPHostUser', max_length=255, blank=True, null=True)
    smtphostpassword = models.CharField(db_column='SMTPHostPassword', max_length=255, blank=True, null=True)
    smtpusetls = models.IntegerField(db_column='SMTPUseTLS', blank=True, null=True)
    trading_experience = models.CharField(db_column='TradingExperience', max_length=20, 
                                        choices=[('beginner', 'Beginner'), ('intermediate', 'Intermediate'), ('advanced', 'Advanced')], 
                                        blank=True, null=True)
    risk_tolerance = models.CharField(db_column='RiskTolerance', max_length=10, 
                                    choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')], 
                                    blank=True, null=True)
    
    isactive = models.IntegerField(blank=True, null=True)
    isdeleted = models.IntegerField(blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)
    createdat = models.DateTimeField(db_column='CreatedAt', blank=True, null=True)
    updatedat = models.DateTimeField(db_column='UpdatedAt', blank=True, null=True)

    @property
    def id(self):
        return self.userid
        
    class Meta:
        managed = True
        db_table = 'Users'
        indexes = [
            models.Index(fields=['email', 'isactive']),
            models.Index(fields=['role', 'isactive']),
        ]
        


# =============================================================================
# TRADING MODELS
# =============================================================================

class Exchange(models.Model):
    exchangeid = models.AutoField(db_column='ExchangeID', primary_key=True)
    name = models.CharField(db_column='Name', max_length=50)  # CCXT ID
    display_name = models.CharField(db_column='DisplayName', max_length=100)
    market_type = models.CharField(db_column='MarketType', max_length=20, blank=True, null=True)
    
    isactive = models.IntegerField(db_column='IsActive', blank=True, null=True)
    isdeleted = models.IntegerField(db_column='IsDeleted', blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)
    created_at = models.DateTimeField(db_column='CreatedAt', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='UpdatedAt', auto_now=True)

    class Meta:
        managed = True
        db_table = 'Exchanges'
        unique_together = ('name', 'market_type')


class TradingPair(models.Model):
    tradingpairid = models.AutoField(db_column='TradingPairID', primary_key=True)
    symbol = models.CharField(db_column='Symbol', max_length=20)
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, db_column='ExchangeID')
    base_currency = models.CharField(db_column='BaseCurrency', max_length=10)
    quote_currency = models.CharField(db_column='QuoteCurrency', max_length=10)

    isactive = models.IntegerField(db_column='IsActive', blank=True, null=True)
    isdeleted = models.IntegerField(db_column='IsDeleted', blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)
    created_at = models.DateTimeField(db_column='CreatedAt', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='UpdatedAt', auto_now=True)

    class Meta:
        managed = True
        db_table = 'TradingPairs'
        unique_together = ('symbol', 'exchange')
        indexes = [
            models.Index(fields=['base_currency']),
            models.Index(fields=['quote_currency']),
        ]

class SeriesConfig(models.Model):
    """Persistent configuration for each time series"""
    seriesconfigid = models.AutoField(db_column='SeriesConfigID', primary_key=True)
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, db_column='ExchangeID')
    market_type = models.CharField(db_column='MarketType', max_length=20)
    symbol = models.CharField(db_column='Symbol', max_length=20)
    timeframe = models.CharField(db_column='Timeframe', max_length=10)
    
    # Core timestamps (all in milliseconds)
    earliest_start_ms = models.BigIntegerField(db_column='EarliestStartMS', null=True, blank=True)
    last_backfill_ms = models.BigIntegerField(db_column='LastBackfillMS', null=True, blank=True)
    last_ts_ms = models.BigIntegerField(db_column='LastTsMS', null=True, blank=True)
    
    # Configuration flags
    probing_completed = models.BooleanField(db_column='ProbingCompleted', default=False)
    backfill_completed = models.BooleanField(db_column='BackfillCompleted', default=False)
    listing_date_ms = models.BigIntegerField(db_column='ListingDateMS', null=True, blank=True)
    
    # Metadata
    probe_attempts = models.IntegerField(db_column='ProbeAttempts', default=0)
    last_error = models.TextField(db_column='LastError', blank=True, null=True)
    
    isactive = models.IntegerField(db_column='IsActive', default=1)
    isdeleted = models.IntegerField(db_column='IsDeleted', default=0)
    created_at = models.DateTimeField(db_column='CreatedAt', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='UpdatedAt', auto_now=True)

    class Meta:
        managed = True
        db_table = 'SeriesConfig'
        unique_together = ('exchange', 'market_type', 'symbol', 'timeframe')
        indexes = [
            models.Index(fields=['exchange', 'market_type', 'symbol', 'timeframe']),
            models.Index(fields=['backfill_completed']),
            models.Index(fields=['probing_completed']),
        ]

    @property
    def series_key(self):
        """Generate series key for caching"""
        return f"{self.exchange.exchangeid}_{self.market_type}_{self.symbol}_{self.timeframe}"

class WorkerShard(models.Model):
    """Track worker sharding configuration"""
    shardid = models.AutoField(db_column='ShardID', primary_key=True)
    shard_name = models.CharField(db_column='ShardName', max_length=50)
    worker_id = models.CharField(db_column='WorkerID', max_length=100)
    
    # Series assignment
    assigned_series = models.JSONField(db_column='AssignedSeries', default=list)
    
    # Status tracking
    status = models.CharField(db_column='Status', max_length=20, default='idle')  # idle, backfilling, streaming, error
    last_heartbeat = models.DateTimeField(db_column='LastHeartbeat', auto_now=True)
    
    # Performance metrics
    series_count = models.IntegerField(db_column='SeriesCount', default=0)
    backfill_progress = models.FloatField(db_column='BackfillProgress', default=0.0)
    
    isactive = models.IntegerField(db_column='IsActive', default=1)
    isdeleted = models.IntegerField(db_column='IsDeleted', default=0)
    created_at = models.DateTimeField(db_column='CreatedAt', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='UpdatedAt', auto_now=True)

    class Meta:
        managed = True
        db_table = 'WorkerShards'
        

class Trendinghashtag(models.Model):
    trendinghashtagid = models.AutoField(db_column='TrendingHashtagID', primary_key=True)
    hashtag = models.CharField(db_column='Hashtag', max_length=100)
    timestamp = models.DateTimeField(db_column='Timestamp', auto_now_add=True)
    count_1h = models.IntegerField(db_column='Count1H', default=0)
    count_6h = models.IntegerField(db_column='Count6H', default=0)
    count_24h = models.IntegerField(db_column='Count24H', default=0)
    velocity = models.FloatField(db_column='Velocity', default=0.0)
    avg_sentiment = models.FloatField(db_column='AvgSentiment', default=0.0)
    trend_score = models.FloatField(db_column='TrendScore', default=0.0)

    class Meta:
        managed = True
        db_table = 'TrendingHashtags'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['hashtag', 'timestamp']),
            models.Index(fields=['trend_score', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.hashtag} @ {self.timestamp}"


class Trendingkeyword(models.Model):
    trendingkeywordid = models.AutoField(db_column='TrendingKeywordID', primary_key=True)
    keyword = models.CharField(db_column='Keyword', max_length=100)
    timestamp = models.DateTimeField(db_column='Timestamp', auto_now_add=True)
    count_1h = models.IntegerField(db_column='Count1H', default=0)
    count_6h = models.IntegerField(db_column='Count6H', default=0)
    count_24h = models.IntegerField(db_column='Count24H', default=0)
    velocity = models.FloatField(db_column='Velocity', default=0.0)
    avg_sentiment = models.FloatField(db_column='AvgSentiment', default=0.0)
    sources = models.JSONField(db_column='Sources', default=dict)  # platform -> count

    class Meta:
        managed = True
        db_table = 'TrendingKeywords'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['keyword', 'timestamp']),
        ]


class Trendingtopic(models.Model):
    trendingtopicid = models.AutoField(db_column='TrendingTopicID', primary_key=True)
    topic_id = models.IntegerField(db_column='TopicID')
    topic_name = models.CharField(db_column='TopicName', max_length=255)
    keywords = models.JSONField(db_column='Keywords', default=list)
    timestamp = models.DateTimeField(db_column='Timestamp', auto_now_add=True)
    document_count = models.IntegerField(db_column='DocumentCount', default=0)
    velocity = models.FloatField(db_column='Velocity', default=0.0)
    avg_sentiment = models.FloatField(db_column='AvgSentiment', default=0.0)
    is_spike = models.BooleanField(db_column='IsSpike', default=False)

    class Meta:
        managed = True
        db_table = 'TrendingTopics'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['topic_id', 'timestamp']),
            models.Index(fields=['is_spike', 'timestamp']),
        ]


class UserPortfolio(models.Model):
    """User's portfolio"""
    portfolioid = models.AutoField(db_column='PortfolioID', primary_key=True)
    user = models.ForeignKey('Users', on_delete=models.CASCADE, db_column='UserID')
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, db_column='ExchangeID')
    currency = models.CharField(db_column='Currency', max_length=10)  # BTC, ETH, USDT, etc.
    total_amount = models.DecimalField(db_column='TotalAmount', max_digits=20, decimal_places=8, default=0)
    available_amount = models.DecimalField(db_column='AvailableAmount', max_digits=20, decimal_places=8, default=0)
    locked_amount = models.DecimalField(db_column='LockedAmount', max_digits=20, decimal_places=8, default=0)
    avg_buy_price = models.DecimalField(db_column='AvgBuyPrice', max_digits=20, decimal_places=8, null=True, blank=True)
    
    isactive = models.IntegerField(db_column='IsActive', blank=True, null=True)
    isdeleted = models.IntegerField(db_column='IsDeleted', blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)
    updated_at = models.DateTimeField(db_column='UpdatedAt', auto_now=True)
    created_at = models.DateTimeField(db_column='CreatedAt', auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'UserPortfolios'
        unique_together = ('user', 'exchange', 'currency')
        indexes = [
            models.Index(fields=['user', 'updated_at']),
        ]
        
        
class UserTrade(models.Model):
    """Track user trades for portfolio management"""
    tradeid = models.AutoField(db_column='TradeID', primary_key=True)
    user = models.ForeignKey('Users', on_delete=models.CASCADE, db_column='UserID')
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, db_column='ExchangeID')
    symbol = models.CharField(db_column='Symbol', max_length=20)
    side = models.CharField(db_column='Side', max_length=4, choices=TRADE_SIDES)
    trade_type = models.CharField(db_column='TradeType', max_length=10, choices=TRADE_TYPES)
    amount = models.DecimalField(db_column='Amount', max_digits=20, decimal_places=8)
    price = models.DecimalField(db_column='Price', max_digits=20, decimal_places=8, null=True, blank=True)
    filled_amount = models.DecimalField(db_column='FilledAmount', max_digits=20, decimal_places=8, default=0)
    filled_price = models.DecimalField(db_column='FilledPrice', max_digits=20, decimal_places=8, null=True, blank=True)
    status = models.CharField(db_column='Status', max_length=10, choices=TRADE_STATUS, default='pending')
    exchange_order_id = models.CharField(db_column='ExchangeOrderID', max_length=100, blank=True)
    fees = models.DecimalField(db_column='Fees', max_digits=20, decimal_places=8, default=0)
    
    isactive = models.IntegerField(db_column='IsActive', blank=True, null=True)
    isdeleted = models.IntegerField(db_column='IsDeleted', blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)
    created_at = models.DateTimeField(db_column='CreatedAt', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='UpdatedAt', auto_now=True)

    class Meta:
        managed = True
        db_table = 'UserTrades'
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['exchange', 'symbol']),
        ]

class UserExchangeCredentials(models.Model):
    """Encrypted storage of user's exchange API credentials"""
    credentialid = models.AutoField(db_column='CredentialID', primary_key=True)
    user = models.ForeignKey('Users', on_delete=models.CASCADE, db_column='UserID')
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, db_column='ExchangeID')
    api_key = models.CharField(db_column='APIKey', max_length=500)  # Encrypted
    api_secret = models.CharField(db_column='APISecret', max_length=500)  # Encrypted
    passphrase = models.CharField(db_column='Passphrase', max_length=500, blank=True)  # For OKX, encrypted
    is_testnet = models.BooleanField(db_column='IsTestnet', default=True)
    
    isactive = models.IntegerField(db_column='IsActive', blank=True, null=True)
    isdeleted = models.IntegerField(db_column='IsDeleted', blank=True, null=True)
    createdby = models.IntegerField(db_column='CreatedBy', blank=True, null=True)
    updatedby = models.IntegerField(db_column='UpdatedBy', blank=True, null=True)
    created_at = models.DateTimeField(db_column='CreatedAt', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='UpdatedAt', auto_now=True)

    class Meta:
        managed = True
        db_table = 'UserExchangeCredentials'
        unique_together = ('user', 'exchange')


class APIUsage(models.Model):
    """Track API usage for billing and rate limiting"""
    apiusageid = models.AutoField(db_column='APIUsageID', primary_key=True)
    user = models.ForeignKey('Users', on_delete=models.CASCADE, db_column='UserID')
    exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE, db_column='ExchangeID')
    endpoint_type = models.CharField(db_column='EndpointType', max_length=50)  # 'market_data', 'trading', 'account'
    request_count = models.IntegerField(db_column='RequestCount', default=1)
    timestamp = models.DateTimeField(db_column='Timestamp', auto_now_add=True)
    
    isactive = models.IntegerField(db_column='IsActive', blank=True, null=True)
    isdeleted = models.IntegerField(db_column='IsDeleted', blank=True, null=True)
    
    class Meta:
        managed = True
        db_table = 'APIUsage'
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['exchange', 'timestamp']),
        ]

class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'
