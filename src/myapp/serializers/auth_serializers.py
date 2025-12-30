from django.utils import timezone
import os
import re
import uuid
from rest_framework import serializers
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings
from django.core.files.base import ContentFile
from myapp.models import Users  
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.exceptions import InvalidToken
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth import authenticate
import base64

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['user_id'] = user.userid
        token['role'] = user.role
        return token

    def validate(self, attrs):
        email = attrs.get('email', None)
        password = attrs.get('password', None)

        if email is None or password is None:
            raise AuthenticationFailed("Email and password are required.")

        try:
            user = Users.objects.get(email=email, isactive=1, isdeleted=0)  # Use 1/0 instead of True/False
        except Users.DoesNotExist:
            raise AuthenticationFailed("Invalid email or password.")

        if not check_password(password, user.passwordhash):
            raise AuthenticationFailed("Invalid email or password.")

        refresh = self.get_token(user)

        return {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
            "user_id": user.userid,
            "role": user.role,
        }

class UserSerializer(serializers.ModelSerializer):
    isactive = serializers.IntegerField(default=1)
    isdeleted = serializers.IntegerField(default=0)
    logo = serializers.CharField(write_only=True, required=False, allow_null=True)  # Add logo field
    logo_url = serializers.SerializerMethodField()

    class Meta:
        model = Users
        fields = [
            'userid', 'fullname', 'email', 'passwordhash', 'role',
            'organization', 'phone', 'address', 'state', 'zipcode', 'country',
            'logo', 'logo_url', 'logo_path',  # Add logo fields
            'useusersmtp', 'smtphost', 'smtpport', 'smtphostuser', 
            'smtphostpassword', 'smtpusetls', 'trading_experience', 'risk_tolerance',
            'isactive', 'isdeleted', 'createdby', 'updatedby', 'createdat', 'updatedat'
        ]
        read_only_fields = ('userid', 'createdat', 'updatedat')

    def get_logo_url(self, obj):
        """Return the complete URL for the logo"""
        if obj.logo_path:
            return f"{settings.MEDIA_URL}{obj.logo_path}"
        return None
    
    def create(self, validated_data):
        validated_data['isactive'] = 1 if validated_data.get('isactive', True) else 0
        validated_data['isdeleted'] = 0 if validated_data.get('isdeleted', False) else 0
        validated_data['createdat'] = timezone.now()
        
        # Handle logo upload
        logo_data = validated_data.pop('logo', None)
        
        # Hash the password if not already hashed
        if validated_data.get('passwordhash') and not validated_data['passwordhash'].startswith('pbkdf2_sha256$'):
            validated_data['passwordhash'] = make_password(validated_data['passwordhash'])
        
        instance = super().create(validated_data)
        
        # Process logo if provided
        if logo_data:
            logo_path = self._handle_logo_upload(logo_data, instance)
            if logo_path:
                instance.logo_path = logo_path
                instance.save()
        
        return instance

    def update(self, instance, validated_data):
        validated_data['updatedat'] = timezone.now()
        
        # Handle logo upload
        logo_data = validated_data.pop('logo', None)
        if logo_data:
            # Delete old logo if exists
            if instance.logo_path:
                old_path = os.path.join(settings.MEDIA_ROOT, instance.logo_path)
                if os.path.exists(old_path):
                    os.remove(old_path)
            
            logo_path = self._handle_logo_upload(logo_data, instance)
            if logo_path:
                validated_data['logo_path'] = logo_path
        
        return super().update(instance, validated_data)
    
    def _handle_logo_upload(self, logo_data, instance):
        """Handle the logo upload process"""
        if not logo_data:
            return None

        try:
            format, imgstr = logo_data.split(';base64,') if ';base64,' in logo_data else ('', logo_data)
            ext = format.split('/')[-1] if format else 'png'
            data = ContentFile(base64.b64decode(imgstr))
            
            # Create user-specific upload path
            relative_path = f'users/{instance.userid}/logo_{uuid.uuid4().hex}.{ext}'
            full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'wb') as f:
                f.write(data.read())
                
            return relative_path

        except Exception as e:
            raise serializers.ValidationError(f"Error processing logo: {str(e)}")
    
    def validate(self, data):
        """Custom validation for user data."""
        # Validate email format and uniqueness
        if 'email' in data:
            from django.core.validators import validate_email
            try:
                validate_email(data['email'])
            except:
                raise serializers.ValidationError({
                    "email": "Invalid email format."
                })
            
            # Check email uniqueness
            if Users.objects.filter(email=data['email'], isdeleted=0).exclude(userid=getattr(self.instance, 'userid', None)).exists():
                raise serializers.ValidationError({
                    "email": "This email is already in use."
                })

        # Validate SMTP settings
        if 'useusersmtp' in data and data.get('useusersmtp') == 1:
            required_smtp_fields = ['smtphost', 'smtpport', 'smtphostuser', 'smtphostpassword']
            for field in required_smtp_fields:
                if not data.get(field):
                    raise serializers.ValidationError({
                        field: f"{field} is required when custom SMTP is enabled."
                    })
            
            # Validate SMTP port range
            if data.get('smtpport') and (data['smtpport'] < 1 or data['smtpport'] > 65535):
                raise serializers.ValidationError({
                    "smtpport": "SMTP port must be between 1 and 65535."
                })

        return data