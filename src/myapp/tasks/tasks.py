from celery import shared_task
from django.utils import timezone
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@shared_task
def cleanup_old_api_usage():
    """Clean up old API usage records"""
    try:
        from myapp.models import APIUsage
         
        # Keep 30 days of data
        cutoff_date = timezone.now() - timedelta(days=30)
        deleted_count = APIUsage.objects.filter(
            timestamp__lt=cutoff_date
        ).delete()[0]
        
        logger.info(f"Cleaned up {deleted_count} old API usage records")
        return f"Cleaned up {deleted_count} old API usage records"
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_api_usage: {e}")
        return f"Error: {e}"