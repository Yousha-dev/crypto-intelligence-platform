from django.core.management.base import BaseCommand
from myapp.services.mongo_manager import get_mongo_manager

class Command(BaseCommand):
    help = 'Clear all content from database'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm deletion'
        )
    
    def handle(self, *args, **options):
        if not options['confirm']:
            self.stdout.write(
                self.style.WARNING('This will delete all content!')
            )
            self.stdout.write('Run with --confirm to proceed')
            return
        
        mongo = get_mongo_manager()
        
        # Delete all documents
        news_count = mongo.collections['news_articles'].delete_many({}).deleted_count
        social_count = mongo.collections['social_posts'].delete_many({}).deleted_count
        
        self.stdout.write(self.style.SUCCESS(
            f'Deleted:\n'
            f'   News articles: {news_count}\n'
            f'   Social posts: {social_count}\n'
        ))