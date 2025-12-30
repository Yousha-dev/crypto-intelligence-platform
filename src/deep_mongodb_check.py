"""
Deep MongoDB Diagnostic - Check actual document structure
Run: python manage.py shell < deep_mongodb_check.py
"""

from myapp.services.mongo_manager import get_mongo_manager
import json

mongo = get_mongo_manager()

# Get one recent article
articles = list(mongo.collections['news_articles'].find().sort('created_at', -1).limit(1))

if articles:
    article = articles[0]
    print("\n" + "="*80)
    print("ACTUAL MONGODB DOCUMENT STRUCTURE")
    print("="*80)
    
    print("FULL DOCUMENT (JSON):")
    print("-"*80)
    # Convert ObjectId to string for JSON serialization
    article['_id'] = str(article['_id'])
    if 'created_at' in article:
        article['created_at'] = str(article['created_at'])
    if 'updated_at' in article:
        article['updated_at'] = str(article['updated_at'])
    print(json.dumps(article, indent=2, default=str))
    
else:
    print("No articles found in MongoDB")

print("\n" + "="*80)



from myapp.services.mongo_manager import get_mongo_manager
import json

mongo = get_mongo_manager()


platforms = ['reddit', 'twitter', 'youtube']
for platform in platforms:
    posts = list(mongo.collections['social_posts'].find({'platform': platform}).sort('created_at', -1).limit(1))

    print("\n" + "="*80)
    print(f"ACTUAL SOCIAL POST DOCUMENT STRUCTURE ({platform.upper()})")
    print("="*80)

    if posts:
        post = posts[0]
        print("\n" + "-"*80)
        print("FULL DOCUMENT (JSON):")
        print("-"*80)
        print(json.dumps(post, indent=2, default=str))
    else:
        print(f"No social posts found for platform: {platform}")

print("\n" + "="*80)

exec(open('deep_mongodb_check.py').read())