"""
Mock News Data Fetcher - Uses local JSON files instead of API calls
Seamlessly replaces the real fetchers with EXACT same signatures and output structure
"""
import os
import json
import time
import re
from datetime import datetime
from django.utils import timezone
from pathlib import Path
from collections import Counter
from textblob import TextBlob

# Get the directory where mock data is stored
MOCK_DATA_DIR = Path(__file__).parent / "mock_news"


def load_mock_data(filename: str) -> list:
    """Load mock data from JSON file"""
    filepath = MOCK_DATA_DIR / filename
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove JavaScript-style comments if present
                lines = content.split('\n')
                cleaned_lines = [line for line in lines if not line.strip().startswith('//')]
                cleaned_content = '\n'.join(cleaned_lines)
                return json.loads(cleaned_content)
        else:
            print(f"️ Mock data file not found: {filepath}")
            return []
    except json.JSONDecodeError as e:
        print(f"Error parsing {filename}: {e}")
        return []
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []


# ============================================================================
# CREDIBILITY ANALYSIS FUNCTIONS (Exact copies from real news_data.py)
# ============================================================================

def analyze_news_source_credibility(item, platform="cryptopanic"):
    """Analyze source credibility for news articles"""
    try:
        source_name = ""
        source_info = {}
        
        if platform == "cryptopanic":
            source_name = item.get("source", {}).get("title", "") if isinstance(item.get("source"), dict) else str(item.get("source", ""))
            source_info = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
        elif platform == "cryptocompare":
            source_name = item.get("source", "")
            source_info = item.get("source_info", {})
        elif platform == "newsapi":
            source_data = item.get("source", {})
            source_name = source_data.get("name", "") if isinstance(source_data, dict) else str(source_data)
            source_info = source_data if isinstance(source_data, dict) else {}
        elif platform == "messari":
            source_name = "Messari"
            source_info = {"platform": "messari"}
        elif platform == "coindesk":
            source_name = "CoinDesk"
            source_info = {"platform": "coindesk"}
        
        high_credibility_sources = {
            "coindesk": 9.0, "cointelegraph": 8.5, "decrypt": 8.0, "the block": 8.5,
            "messari": 9.0, "coinbase": 7.5, "binance": 7.0, "forbes": 8.0,
            "reuters": 9.5, "bloomberg": 9.0, "wall street journal": 9.0, "financial times": 8.5
        }
        
        medium_credibility_sources = {
            "bitcoin magazine": 7.5, "cryptoslate": 7.0, "newsbtc": 6.5,
            "cryptopotato": 6.0, "u.today": 6.0, "ambcrypto": 6.0
        }
        
        source_lower = source_name.lower()
        trust_score = 5.0
        source_tier = "Unknown"
        
        for source, score in high_credibility_sources.items():
            if source in source_lower:
                trust_score = score
                source_tier = "High Credibility"
                break
        else:
            for source, score in medium_credibility_sources.items():
                if source in source_lower:
                    trust_score = score
                    source_tier = "Medium Credibility"
                    break
            else:
                if any(keyword in source_lower for keyword in ["official", "gov", ".gov", "reuters", "ap news"]):
                    trust_score = 8.5
                    source_tier = "Official Source"
                elif any(keyword in source_lower for keyword in ["blog", "personal", "unknown"]):
                    trust_score = 3.0
                    source_tier = "Low Credibility"
        
        credibility_indicators = {
            "is_known_source": trust_score > 5.0,
            "is_high_credibility": trust_score >= 8.0,
            "is_official_source": source_tier == "Official Source",
            "has_source_info": bool(source_info),
            "source_name_length": len(source_name),
            "appears_professional": not bool(re.search(r'[0-9]{3,}|xxx|pump|moon|rocket', source_lower))
        }
        
        return {
            "source_name": source_name,
            "source_info": source_info,
            "trust_score": trust_score,
            "source_tier": source_tier,
            "credibility_indicators": credibility_indicators,
            "platform": platform
        }
        
    except Exception as e:
        return {"source_name": "Unknown", "trust_score": 3.0, "source_tier": "Unknown", "credibility_indicators": {}, "error": str(e)}


def analyze_news_content_quality(item, platform="cryptopanic"):
    """Analyze content quality for news articles"""
    try:
        title = ""
        content = ""
        description = ""
        
        if platform == "cryptopanic":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
            description = item.get("description") or ""
        elif platform == "cryptocompare":
            title = item.get("title") or ""
            content = item.get("body") or ""
            description = content[:200] if content else ""
        elif platform == "newsapi":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
            description = item.get("description") or ""
        elif platform == "messari":
            title = item.get("title") or ""
            content = item.get("content") or ""
            description = content[:200] if content else ""
        elif platform == "coindesk":
            title = getattr(item, "title", "") or item.get("title", "") or ""
            content = getattr(item, "summary", "") or item.get("summary", "") or item.get("content", "") or ""
            description = content
        
        title = str(title) if title else ""
        content = str(content) if content else ""
        description = str(description) if description else ""
        all_text = f"{title} {description} {content}"
        
        quality_factors = {
            "has_meaningful_title": len(title) > 10 and len(title) < 200,
            "has_content": len(content) > 50,
            "has_description": len(description) > 20,
            "appropriate_length": 100 < len(all_text) < 5000,
            "not_all_caps": not title.isupper() if title else True,
            "no_excessive_punctuation": len(re.findall(r'[!?]{2,}', all_text)) < 3,
            "proper_grammar": not bool(re.search(r'\b(ur|u|pls|omg|lol|rofl)\b', all_text.lower())),
            "no_spam_keywords": not bool(re.search(r'\b(click here|buy now|limited time|act fast|guaranteed|100%)\b', all_text.lower())),
            "professional_tone": not bool(re.search(r'🚀|💰|📈|MOON|PUMP|DUMP', all_text))
        }
        
        if all_text.strip():
            word_count = len(all_text.split())
            sentence_count = len(re.split(r'[.!?]+', all_text))
            avg_sentence_length = word_count / max(sentence_count, 1)
        else:
            word_count = 0
            sentence_count = 0
            avg_sentence_length = 0
        
        quality_score = sum(quality_factors.values()) / len(quality_factors) * 10
        
        readability_factors = {
            "appropriate_word_count": 50 < word_count < 1000,
            "good_sentence_length": 10 < avg_sentence_length < 25,
            "has_structure": bool(re.search(r'[.!?]', all_text)),
            "varied_vocabulary": len(set(all_text.lower().split())) / max(word_count, 1) > 0.5 if word_count > 0 else False
        }
        
        readability_score = sum(readability_factors.values()) / len(readability_factors) * 10
        final_quality_score = (quality_score * 0.7) + (readability_score * 0.3)
        
        tier = "Unknown"
        if final_quality_score >= 8.5:
            tier = "Excellent Quality"
        elif final_quality_score >= 7.0:
            tier = "High Quality"
        elif final_quality_score >= 5.5:
            tier = "Good Quality"
        elif final_quality_score >= 4.0:
            tier = "Fair Quality"
        else:
            tier = "Poor Quality"
        
        return {
            "quality_factors": quality_factors,
            "quality_score": quality_score,
            "readability_factors": readability_factors,
            "readability_score": readability_score,
            "final_quality_score": final_quality_score,
            "content_stats": {
                "title_length": len(title),
                "content_length": len(content),
                "total_length": len(all_text),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length
            },
            "quality_tier": tier
        }
        
    except Exception as e:
        return {"final_quality_score": 5.0, "quality_tier": "Unknown", "error": str(e)}


def analyze_news_sentiment(item, platform="cryptopanic"):
    """Analyze sentiment of news articles"""
    try:
        title = ""
        content = ""
        
        if platform == "cryptopanic":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
        elif platform == "cryptocompare":
            title = item.get("title") or ""
            content = item.get("body") or ""
        elif platform == "newsapi":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
        elif platform == "messari":
            title = item.get("title") or ""
            content = item.get("content") or ""
        elif platform == "coindesk":
            title = getattr(item, "title", "") or item.get("title", "") or ""
            content = getattr(item, "summary", "") or item.get("summary", "") or item.get("content", "") or ""
        
        title = str(title) if title else ""
        content = str(content) if content else ""
        all_text = f"{title} {content}".strip()
        
        if not all_text:
            return {"sentiment_label": "Neutral", "crypto_sentiment_score": 0.0, "confidence": 0.0}
        
        try:
            blob = TextBlob(all_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except:
            polarity = 0.0
            subjectivity = 0.5
        
        bullish_keywords = ["bullish", "bull", "rise", "up", "gain", "positive", "growth", "increase", "surge", "rally", "moon", "pump"]
        bearish_keywords = ["bearish", "bear", "fall", "down", "loss", "negative", "decline", "decrease", "crash", "dump", "drop"]
        neutral_keywords = ["stable", "consolidate", "sideways", "hold", "range", "analyze", "technical"]
        
        text_lower = all_text.lower()
        bullish_count = sum(text_lower.count(word) for word in bullish_keywords)
        bearish_count = sum(text_lower.count(word) for word in bearish_keywords)
        neutral_count = sum(text_lower.count(word) for word in neutral_keywords)
        
        total_sentiment_words = bullish_count + bearish_count + neutral_count
        if total_sentiment_words > 0:
            crypto_sentiment = (bullish_count - bearish_count) / total_sentiment_words
        else:
            crypto_sentiment = polarity
        
        if crypto_sentiment > 0.2:
            sentiment_label = "Bullish"
        elif crypto_sentiment < -0.2:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"
        
        confidence = min(abs(crypto_sentiment) + (subjectivity * 0.3), 1.0)
        
        # Sentiment strength
        abs_score = abs(crypto_sentiment)
        if abs_score >= 0.7:
            strength = "Very Strong"
        elif abs_score >= 0.5:
            strength = "Strong"
        elif abs_score >= 0.3:
            strength = "Moderate"
        elif abs_score >= 0.1:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        # Market emotion
        if crypto_sentiment > 0.5 and subjectivity > 0.7:
            emotion = "FOMO (Fear of Missing Out)"
        elif crypto_sentiment < -0.5 and subjectivity > 0.7:
            emotion = "FUD (Fear, Uncertainty, Doubt)"
        elif abs(crypto_sentiment) < 0.2:
            emotion = "Calm/Analytical"
        elif crypto_sentiment > 0:
            emotion = "Optimistic"
        else:
            emotion = "Pessimistic"
        
        return {
            "textblob_polarity": polarity,
            "textblob_subjectivity": subjectivity,
            "crypto_sentiment_score": crypto_sentiment,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "keyword_analysis": {
                "bullish_keywords": bullish_count,
                "bearish_keywords": bearish_count,
                "neutral_keywords": neutral_count,
                "total_sentiment_words": total_sentiment_words
            },
            "sentiment_strength": strength,
            "market_emotion": emotion
        }
        
    except Exception as e:
        return {"sentiment_label": "Neutral", "crypto_sentiment_score": 0.0, "confidence": 0.0, "error": str(e)}


def analyze_crypto_relevance_news(item, platform="cryptopanic"):
    """Analyze crypto relevance of news articles"""
    try:
        title = ""
        content = ""
        tags = []
        categories = []
        
        if platform == "cryptopanic":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
            instruments = item.get("instruments", [])
            if instruments:
                categories = [instr.get("code", "") for instr in instruments if isinstance(instr, dict)]
        elif platform == "cryptocompare":
            title = item.get("title", "")
            content = item.get("body", "")
            tags = item.get("tags", "").split(",") if item.get("tags") else []
            categories = item.get("categories", "").split(",") if item.get("categories") else []
        elif platform == "newsapi":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
        elif platform == "messari":
            title = item.get("title", "")
            content = item.get("content", "")
            tags = item.get("tags", [])
        elif platform == "coindesk":
            title = getattr(item, "title", "") or item.get("title", "")
            content = getattr(item, "summary", "") or item.get("summary", "") or item.get("content", "")
        
        all_text = f"{title} {content}".lower()
        
        primary_crypto_keywords = [
            "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto", "blockchain",
            "altcoin", "defi", "nft", "web3", "smart contract", "mining", "wallet"
        ]
        secondary_crypto_keywords = [
            "trading", "exchange", "binance", "coinbase", "market cap", "bull run", "bear market",
            "hodl", "pump", "dump", "moon", "diamond hands", "paper hands", "whale", "satoshi"
        ]
        major_cryptos = [
            "ada", "cardano", "sol", "solana", "matic", "polygon", "dot", "polkadot",
            "link", "chainlink", "uni", "uniswap", "aave", "comp", "compound", "maker", "mkr"
        ]
        
        primary_matches = sum(all_text.count(keyword) for keyword in primary_crypto_keywords)
        secondary_matches = sum(all_text.count(keyword) for keyword in secondary_crypto_keywords)
        crypto_mentions = sum(all_text.count(crypto) for crypto in major_cryptos)
        
        relevance_score = (
            primary_matches * 3 +
            secondary_matches * 2 +
            crypto_mentions * 1 +
            len(categories) * 2 +
            len(tags) * 0.5
        )
        
        final_relevance_score = min(relevance_score / 5, 10.0)
        
        focus_areas = []
        if any(keyword in all_text for keyword in ["trading", "market", "price", "bull", "bear"]):
            focus_areas.append("Trading/Markets")
        if any(keyword in all_text for keyword in ["defi", "smart contract", "protocol"]):
            focus_areas.append("DeFi/Technology")
        if any(keyword in all_text for keyword in ["nft", "art", "collectible"]):
            focus_areas.append("NFTs")
        if any(keyword in all_text for keyword in ["regulation", "sec", "government", "legal"]):
            focus_areas.append("Regulation")
        if any(keyword in all_text for keyword in ["adoption", "institutional", "company"]):
            focus_areas.append("Adoption")
        
        # Extract mentioned cryptos
        crypto_patterns = {
            "Bitcoin": r"\b(bitcoin|btc)\b",
            "Ethereum": r"\b(ethereum|eth)\b",
            "Cardano": r"\b(cardano|ada)\b",
            "Solana": r"\b(solana|sol)\b",
            "Polkadot": r"\b(polkadot|dot)\b",
            "Chainlink": r"\b(chainlink|link)\b",
            "Polygon": r"\b(polygon|matic)\b",
            "Uniswap": r"\b(uniswap|uni)\b"
        }
        mentioned = [crypto for crypto, pattern in crypto_patterns.items() if re.search(pattern, all_text, re.IGNORECASE)]
        
        # Relevance tier
        if final_relevance_score >= 8:
            tier = "Highly Relevant"
        elif final_relevance_score >= 6:
            tier = "Very Relevant"
        elif final_relevance_score >= 4:
            tier = "Relevant"
        elif final_relevance_score >= 2:
            tier = "Somewhat Relevant"
        else:
            tier = "Not Relevant"
        
        return {
            "relevance_score": final_relevance_score,
            "primary_keyword_matches": primary_matches,
            "secondary_keyword_matches": secondary_matches,
            "crypto_mentions": crypto_mentions,
            "categories_mentioned": categories,
            "tags": tags,
            "focus_areas": focus_areas,
            "is_crypto_focused": final_relevance_score > 3.0,
            "relevance_tier": tier,
            "mentioned_cryptocurrencies": mentioned
        }
        
    except Exception as e:
        return {"relevance_score": 5.0, "is_crypto_focused": True, "relevance_tier": "Unknown", "error": str(e)}


def analyze_market_impact_potential(item, platform="cryptopanic"):
    """Analyze potential market impact of news"""
    try:
        title = ""
        content = ""
        votes = 0
        panic_score = 0
        
        if platform == "cryptopanic":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
            votes = item.get("votes", {}).get("positive", 0) if isinstance(item.get("votes"), dict) else 0
            panic_score = item.get("panic_score", 0) or 0
        elif platform == "cryptocompare":
            title = item.get("title", "")
            content = item.get("body", "")
            upvotes = item.get("upvotes", 0)
            downvotes = item.get("downvotes", 0)
            try:
                upvotes = int(str(upvotes)) if upvotes else 0
                downvotes = int(str(downvotes)) if downvotes else 0
                votes = upvotes - downvotes
            except:
                votes = 0
        elif platform == "newsapi":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
        elif platform == "messari":
            title = item.get("title", "")
            content = item.get("content", "")
        elif platform == "coindesk":
            title = getattr(item, "title", "") or item.get("title", "")
            content = getattr(item, "summary", "") or item.get("summary", "") or item.get("content", "")
        
        title = str(title) if title else ""
        content = str(content) if content else ""
        all_text = f"{title} {content}".lower()
        
        high_impact_keywords = [
            "breaking", "urgent", "major", "significant", "important", "critical",
            "sec", "regulation", "ban", "approval", "etf", "institutional",
            "hack", "security", "breach", "crash", "surge", "rally"
        ]
        market_events = [
            "etf approval", "sec decision", "regulatory", "institutional adoption",
            "major partnership", "listing", "delisting", "hack", "upgrade",
            "hard fork", "merger", "acquisition"
        ]
        
        high_impact_count = sum(all_text.count(keyword) for keyword in high_impact_keywords)
        market_event_count = sum(all_text.count(event) for event in market_events)
        
        impact_score = high_impact_count * 2 + market_event_count * 3
        
        if platform == "cryptopanic" and panic_score:
            try:
                impact_score += float(panic_score) / 20
            except:
                pass
        
        if votes > 0:
            impact_score += min(votes / 10, 2)
        
        if any(source in all_text for source in ["reuters", "bloomberg", "sec", "official"]):
            impact_score += 2
        
        final_impact_score = min(impact_score, 10.0)
        
        impact_factors = {
            "has_breaking_news": "breaking" in all_text,
            "regulatory_news": any(word in all_text for word in ["sec", "regulation", "regulatory"]),
            "institutional_news": "institutional" in all_text,
            "technical_news": any(word in all_text for word in ["upgrade", "fork", "protocol"]),
            "market_data": any(word in all_text for word in ["price", "volume", "market cap"]),
            "high_engagement": votes > 50 if votes else False,
            "panic_indicator": panic_score > 50 if panic_score else False
        }
        
        # Impact tier
        if final_impact_score >= 8:
            tier = "High Impact"
        elif final_impact_score >= 6:
            tier = "Medium-High Impact"
        elif final_impact_score >= 4:
            tier = "Medium Impact"
        elif final_impact_score >= 2:
            tier = "Low-Medium Impact"
        else:
            tier = "Low Impact"
        
        # Market effect
        if final_impact_score >= 8:
            if impact_factors.get("regulatory_news"):
                effect = "Major Regulatory Impact"
            elif impact_factors.get("institutional_news"):
                effect = "Institutional Movement"
            else:
                effect = "Significant Market Moving"
        elif final_impact_score >= 6:
            effect = "Moderate Market Influence"
        elif final_impact_score >= 4:
            effect = "Minor Market Effect"
        else:
            effect = "Limited Market Impact"
        
        return {
            "impact_score": final_impact_score,
            "high_impact_keywords": high_impact_count,
            "market_events": market_event_count,
            "votes": votes,
            "panic_score": panic_score,
            "impact_factors": impact_factors,
            "impact_tier": tier,
            "potential_market_effect": effect
        }
        
    except Exception as e:
        return {"impact_score": 3.0, "impact_tier": "Low Impact", "error": str(e)}


def calculate_news_trust_score(item, source_credibility, content_analysis, 
                                    sentiment_analysis, crypto_analysis, market_impact):
    """Calculate overall credibility score for news articles"""
    try:
        source_weight = 0.35
        content_weight = 0.25
        sentiment_weight = 0.15
        crypto_weight = 0.15
        impact_weight = 0.10
        
        source_score = source_credibility.get('trust_score', 5.0)
        content_score = content_analysis.get('final_quality_score', 5.0)
        sentiment_confidence = sentiment_analysis.get('confidence', 0.5) * 10
        crypto_score = crypto_analysis.get('relevance_score', 5.0)
        impact_score = market_impact.get('impact_score', 3.0)
        
        final_score = (
            source_score * source_weight +
            content_score * content_weight +
            sentiment_confidence * sentiment_weight +
            crypto_score * crypto_weight +
            impact_score * impact_weight
        )
        
        final_score = max(0.0, min(final_score, 10.0))
        
        credibility_factors = {
            "high_credibility_source": source_credibility.get('trust_score', 0) >= 8.0,
            "good_content_quality": content_analysis.get('final_quality_score', 0) >= 7.0,
            "clear_sentiment": sentiment_analysis.get('confidence', 0) >= 0.7,
            "crypto_relevant": crypto_analysis.get('is_crypto_focused', False),
            "high_market_impact": market_impact.get('impact_score', 0) >= 7.0,
            "professional_source": source_credibility.get('credibility_indicators', {}).get('appears_professional', False),
            "substantial_content": content_analysis.get('content_stats', {}).get('word_count', 0) > 100
        }
        
        # Credibility tier
        if final_score >= 8.5:
            tier = "Highly Credible"
        elif final_score >= 7.0:
            tier = "Very Credible"
        elif final_score >= 5.5:
            tier = "Credible"
        elif final_score >= 4.0:
            tier = "Moderately Credible"
        elif final_score >= 2.5:
            tier = "Low Credibility"
        else:
            tier = "Very Low Credibility"
        
        # Recommendation
        if final_score >= 8.0:
            recommendation = "Highly Recommended - Excellent source"
        elif final_score >= 7.0:
            recommendation = "Recommended - Good credibility"
        elif final_score >= 5.0:
            recommendation = "Acceptable - Verify with other sources"
        elif final_score >= 3.0:
            recommendation = "Use Caution - Low credibility"
        else:
            recommendation = "Not Recommended - Very low credibility"
        
        return {
            "final_trust_score": final_score,
            "score_breakdown": {
                "source_credibility": source_score,
                "content_quality": content_score,
                "sentiment_confidence": sentiment_confidence,
                "crypto_relevance": crypto_score,
                "market_impact": impact_score
            },
            "weights_used": {
                "source": source_weight,
                "content": content_weight,
                "sentiment": sentiment_weight,
                "crypto": crypto_weight,
                "impact": impact_weight
            },
            "credibility_factors": credibility_factors,
            "credibility_tier": tier,
            "recommendation": recommendation
        }
        
    except Exception as e:
        return {"final_trust_score": 5.0, "credibility_tier": "Medium Credibility", "error": str(e)}


def analyze_news_results(news_articles, platform_name):
    """Analyze and summarize news results"""
    if not news_articles:
        return
    
    total_articles = len(news_articles)
    avg_credibility = sum(
        article.get('overall_credibility', {}).get('final_trust_score', 0) 
        for article in news_articles
    ) / total_articles
    
    tiers = {
        "Highly Credible": 0, "Very Credible": 0, "Credible": 0, 
        "Moderately Credible": 0, "Low Credibility": 0, "Very Low Credibility": 0
    }
    
    for article in news_articles:
        tier = article.get('overall_credibility', {}).get('credibility_tier', 'Very Low Credibility')
        if tier in tiers:
            tiers[tier] += 1
    
    sentiments = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    for article in news_articles:
        sentiment = article.get('sentiment_analysis', {}).get('sentiment_label', 'Neutral')
        if sentiment in sentiments:
            sentiments[sentiment] += 1
    
    print(f"📈 [MOCK] {platform_name} Analysis Summary:")
    print(f"   Total Articles: {total_articles}")
    print(f"   Average Credibility: {avg_credibility:.2f}/10")
    print(f"   Sentiment Distribution: {dict(list(sentiments.items()))}")
    print(f"   Top Credibility Tiers: {dict(list(tiers.items())[:3])}")


# ============================================================================
# MOCK FETCHER FUNCTIONS (EXACT same signatures as real fetchers)
# ============================================================================

def fetch_cryptopanic_news_enhanced(currencies=None, filter_type=None, kind="news", 
                                   size=None, search=None, analyze_credibility=True, 
                                   max_items=100):
    """
    MOCK: Fetch news from CryptoPanic API with comprehensive credibility and sentiment analysis
    EXACT same signature and output structure as real fetcher
    """
    print(f"[MOCK] Fetching CryptoPanic news with credibility analysis")
    print(f"Loading from: {MOCK_DATA_DIR / 'cryptopanic.json'}")
    print(f"API Budget: Processing up to {max_items} items")
    
    time.sleep(0.3)  # Simulate API delay
    
    articles = load_mock_data("cryptopanic.json")
    
    if not articles:
        print("️ No mock CryptoPanic data found")
        return []
    
    enhanced_news = []
    
    for item in articles[:max_items]:
        try:
            article = {
                # Original fields (exact match to real fetcher)
                "id": item.get("id"),
                "slug": item.get("slug"),
                "title": item.get("title"),
                "description": item.get("description"),
                "url": item.get("url"),
                "original_url": item.get("original_url"),
                "published_at": item.get("published_at"),
                "created_at": item.get("created_at"),
                "kind": item.get("kind", kind),
                "source": item.get("source"),
                "image": item.get("image"),
                "instruments": item.get("instruments", []),
                "votes": item.get("votes"),
                "panic_score": item.get("panic_score"),
                "panic_score_1h": item.get("panic_score_1h"),
                "author": item.get("author"),
                "content": item.get("content"),
                
                # Enhanced fields
                "platform": "CryptoPanic",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if analyze_credibility:
                source_credibility = analyze_news_source_credibility(item, "cryptopanic")
                content_analysis = analyze_news_content_quality(item, "cryptopanic")
                sentiment_analysis = analyze_news_sentiment(item, "cryptopanic")
                crypto_analysis = analyze_crypto_relevance_news(item, "cryptopanic")
                market_impact = analyze_market_impact_potential(item, "cryptopanic")
                
                overall_credibility = calculate_news_trust_score(
                    item, source_credibility, content_analysis,
                    sentiment_analysis, crypto_analysis, market_impact
                )
                
                article.update({
                    "source_credibility": source_credibility,
                    "content_analysis": content_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "crypto_analysis": crypto_analysis,
                    "market_impact": market_impact,
                    "overall_credibility": overall_credibility
                })
            
            enhanced_news.append(article)
            
        except Exception as e:
            print(f"Error processing CryptoPanic item {item.get('id')}: {e}")
            continue
    
    if analyze_credibility and enhanced_news:
        enhanced_news.sort(
            key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
            reverse=True
        )
        analyze_news_results(enhanced_news, "CryptoPanic")
    
    print(f"[MOCK] CryptoPanic: Processed {len(enhanced_news)} articles")
    
    return enhanced_news


def fetch_cryptocompare_news_enhanced(categories=None, excludeCategories=None, feeds=None,
                                     lTs=None, sortOrder="latest", lang="EN", 
                                     analyze_credibility=True, max_items=100):
    """
    MOCK: Fetch news from CryptoCompare API with enhanced credibility analysis
    EXACT same signature and output structure as real fetcher
    """
    print(f"[MOCK] Fetching CryptoCompare news with credibility analysis")
    print(f"Loading from: {MOCK_DATA_DIR / 'cryptocompare.json'}")
    
    time.sleep(0.3)
    
    articles = load_mock_data("cryptocompare.json")
    
    if not articles:
        print("️ No mock CryptoCompare data found")
        return []
    
    enhanced_news = []
    
    for item in articles[:max_items]:
        try:
            article = {
                # Original fields (exact match)
                "id": item.get("id"),
                "guid": item.get("guid"),
                "published_on": item.get("published_on"),
                "imageurl": item.get("imageurl"),
                "title": item.get("title"),
                "url": item.get("url"),
                "source": item.get("source"),
                "body": item.get("body"),
                "tags": item.get("tags"),
                "categories": item.get("categories"),
                "upvotes": item.get("upvotes"),
                "downvotes": item.get("downvotes"),
                "lang": item.get("lang", lang),
                "source_info": item.get("source_info"),
                
                # Enhanced fields
                "platform": "CryptoCompare",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if analyze_credibility:
                source_credibility = analyze_news_source_credibility(item, "cryptocompare")
                content_analysis = analyze_news_content_quality(item, "cryptocompare")
                sentiment_analysis = analyze_news_sentiment(item, "cryptocompare")
                crypto_analysis = analyze_crypto_relevance_news(item, "cryptocompare")
                market_impact = analyze_market_impact_potential(item, "cryptocompare")
                
                overall_credibility = calculate_news_trust_score(
                    item, source_credibility, content_analysis,
                    sentiment_analysis, crypto_analysis, market_impact
                )
                
                article.update({
                    "source_credibility": source_credibility,
                    "content_analysis": content_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "crypto_analysis": crypto_analysis,
                    "market_impact": market_impact,
                    "overall_credibility": overall_credibility
                })
            
            enhanced_news.append(article)
            
        except Exception as e:
            print(f"Error processing CryptoCompare item {item.get('id')}: {e}")
            continue
    
    if analyze_credibility and enhanced_news:
        enhanced_news.sort(
            key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
            reverse=True
        )
        analyze_news_results(enhanced_news, "CryptoCompare")
    
    print(f"[MOCK] CryptoCompare: Processed {len(enhanced_news)} articles")
    
    return enhanced_news


def fetch_newsapi_articles_enhanced(query="cryptocurrency", page_size=20, 
                                   analyze_credibility=True, max_items=50):
    """
    MOCK: Fetch articles from NewsAPI with enhanced credibility analysis
    EXACT same signature and output structure as real fetcher
    """
    print(f"[MOCK] Fetching NewsAPI articles with credibility analysis")
    print(f"Loading from: {MOCK_DATA_DIR / 'newsapi.json'}")
    
    time.sleep(0.3)
    
    articles = load_mock_data("newsapi.json")
    
    if not articles:
        print("️ No mock NewsAPI data found")
        return []
    
    enhanced_articles = []
    
    for article_data in articles[:max_items]:
        try:
            article = {
                # Original fields (exact match)
                "source": article_data.get("source"),
                "author": article_data.get("author"),
                "title": article_data.get("title"),
                "description": article_data.get("description"),
                "url": article_data.get("url"),
                "urlToImage": article_data.get("urlToImage"),
                "publishedAt": article_data.get("publishedAt"),
                "content": article_data.get("content"),
                
                # Enhanced fields
                "platform": "NewsAPI",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if analyze_credibility:
                source_credibility = analyze_news_source_credibility(article_data, "newsapi")
                content_analysis = analyze_news_content_quality(article_data, "newsapi")
                sentiment_analysis = analyze_news_sentiment(article_data, "newsapi")
                crypto_analysis = analyze_crypto_relevance_news(article_data, "newsapi")
                market_impact = analyze_market_impact_potential(article_data, "newsapi")
                
                overall_credibility = calculate_news_trust_score(
                    article_data, source_credibility, content_analysis,
                    sentiment_analysis, crypto_analysis, market_impact
                )
                
                article.update({
                    "source_credibility": source_credibility,
                    "content_analysis": content_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "crypto_analysis": crypto_analysis,
                    "market_impact": market_impact,
                    "overall_credibility": overall_credibility
                })
            
            enhanced_articles.append(article)
            
        except Exception as e:
            print(f"Error processing NewsAPI article: {e}")
            continue
    
    if analyze_credibility and enhanced_articles:
        enhanced_articles.sort(
            key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
            reverse=True
        )
        analyze_news_results(enhanced_articles, "NewsAPI")
    
    print(f"[MOCK] NewsAPI: Processed {len(enhanced_articles)} articles")
    
    return enhanced_articles


def fetch_messari_news_enhanced(fields=None, limit=50, analyze_credibility=True, max_items=50):
    """
    MOCK: Fetch news from Messari with enhanced analysis
    EXACT same signature and output structure as real fetcher
    """
    print(f"[MOCK] Fetching Messari news with credibility analysis")
    print(f"Loading from: {MOCK_DATA_DIR / 'messari.json'}")
    
    time.sleep(0.3)
    
    articles = load_mock_data("messari.json")
    
    if not articles:
        print("️ No mock Messari data found")
        return []
    
    enhanced_news = []
    
    for item in articles[:max_items]:
        try:
            article = {
                # Original fields (exact match)
                "id": item.get("id"),
                "title": item.get("title"),
                "content": item.get("content"),
                "references": item.get("references", []),
                "reference_title": item.get("reference_title"),
                "published_at": item.get("published_at"),
                "author": item.get("author", {}).get("name") if isinstance(item.get("author"), dict) else item.get("author"),
                "tags": item.get("tags", []),
                "url": item.get("url"),
                
                # Enhanced fields
                "platform": "Messari",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if analyze_credibility:
                source_credibility = analyze_news_source_credibility(item, "messari")
                content_analysis = analyze_news_content_quality(item, "messari")
                sentiment_analysis = analyze_news_sentiment(item, "messari")
                crypto_analysis = analyze_crypto_relevance_news(item, "messari")
                market_impact = analyze_market_impact_potential(item, "messari")
                
                overall_credibility = calculate_news_trust_score(
                    item, source_credibility, content_analysis,
                    sentiment_analysis, crypto_analysis, market_impact
                )
                
                article.update({
                    "source_credibility": source_credibility,
                    "content_analysis": content_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "crypto_analysis": crypto_analysis,
                    "market_impact": market_impact,
                    "overall_credibility": overall_credibility
                })
            
            enhanced_news.append(article)
            
        except Exception as e:
            print(f"Error processing Messari item {item.get('id')}: {e}")
            continue
    
    if analyze_credibility and enhanced_news:
        enhanced_news.sort(
            key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
            reverse=True
        )
        analyze_news_results(enhanced_news, "Messari")
    
    print(f"[MOCK] Messari: Processed {len(enhanced_news)} articles")
    
    return enhanced_news


def scrape_coindesk_enhanced(analyze_credibility=True, max_items=50):
    """
    MOCK: Scrape CoinDesk RSS feed with enhanced analysis
    EXACT same signature and output structure as real fetcher
    """
    print(f"[MOCK] Fetching CoinDesk RSS with credibility analysis")
    print(f"Loading from: {MOCK_DATA_DIR / 'coindesk.json'}")
    
    time.sleep(0.3)
    
    articles = load_mock_data("coindesk.json")
    
    if not articles:
        print("️ No mock CoinDesk data, using fallback...")
        articles = load_mock_data("cryptopanic.json")[:3]
        for article in articles:
            article['source'] = {"title": "CoinDesk"}
    
    enhanced_articles = []
    
    for entry in articles[:max_items]:
        try:
            article = {
                # Original fields (exact match to RSS parser output)
                "title": entry.get("title"),
                "link": entry.get("link") or entry.get("url"),
                "published": entry.get("published") or entry.get("published_at"),
                "summary": entry.get("summary") or entry.get("description") or entry.get("content"),
                "media_content": entry.get("media_content", []),
                
                # Enhanced fields
                "platform": "CoinDesk",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if analyze_credibility:
                source_credibility = analyze_news_source_credibility(entry, "coindesk")
                content_analysis = analyze_news_content_quality(entry, "coindesk")
                sentiment_analysis = analyze_news_sentiment(entry, "coindesk")
                crypto_analysis = analyze_crypto_relevance_news(entry, "coindesk")
                market_impact = analyze_market_impact_potential(entry, "coindesk")
                
                overall_credibility = calculate_news_trust_score(
                    entry, source_credibility, content_analysis,
                    sentiment_analysis, crypto_analysis, market_impact
                )
                
                article.update({
                    "source_credibility": source_credibility,
                    "content_analysis": content_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "crypto_analysis": crypto_analysis,
                    "market_impact": market_impact,
                    "overall_credibility": overall_credibility
                })
            
            enhanced_articles.append(article)
            
        except Exception as e:
            print(f"Error processing CoinDesk entry: {e}")
            continue
    
    if analyze_credibility and enhanced_articles:
        enhanced_articles.sort(
            key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
            reverse=True
        )
        analyze_news_results(enhanced_articles, "CoinDesk")
    
    print(f"[MOCK] CoinDesk: Processed {len(enhanced_articles)} articles")
    
    return enhanced_articles


# ============================================================================
# MAIN FUNCTIONS (EXACT same as real fetchers)
# ============================================================================

def fetch_all_news_enhanced(max_items_per_source=50, trust_score_threshold=6.0, analyze_all=True):
    """
    MOCK: Fetch news from all sources with enhanced analysis
    EXACT same signature and output structure as real fetcher
    """
    print("🚀 [MOCK] Starting Enhanced Multi-Source News Fetching")
    print(f"Processing up to {max_items_per_source} items per source")
    print(f"Credibility threshold: {trust_score_threshold}")
    print("=" * 60)
    
    start_time = time.time()
    all_results = {}
    
    sources = [
        ("cryptopanic", lambda: fetch_cryptopanic_news_enhanced(
            filter_type="important", 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )),
        ("cryptocompare", lambda: fetch_cryptocompare_news_enhanced(
            categories="BTC,ETH", 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )),
        ("newsapi", lambda: fetch_newsapi_articles_enhanced(
            query="cryptocurrency bitcoin ethereum", 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )),
        ("messari", lambda: fetch_messari_news_enhanced(
            limit=max_items_per_source, 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )),
        ("coindesk", lambda: scrape_coindesk_enhanced(
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        ))
    ]
    
    for source_name, fetch_function in sources:
        try:
            print(f"\n🔄 [MOCK] Fetching from {source_name}...")
            articles = fetch_function()
            
            if analyze_all and articles:
                high_credibility_articles = [
                    article for article in articles
                    if article.get('overall_credibility', {}).get('final_trust_score', 0) >= trust_score_threshold
                ]
                
                print(f"{source_name}: {len(high_credibility_articles)}/{len(articles)} articles meet credibility threshold")
                all_results[source_name.lower()] = high_credibility_articles
            else:
                all_results[source_name.lower()] = articles
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            all_results[source_name.lower()] = []
            continue
    
    execution_time = time.time() - start_time
    
    # Generate summary
    total_articles = sum(len(articles) for articles in all_results.values())
    
    print(f"\n[MOCK] ENHANCED NEWS ANALYSIS SUMMARY")
    print("=" * 50)
    print(f" Execution Time: {execution_time:.1f} seconds")
    print(f"Total High-Quality Articles: {total_articles}")
    print(f"🎚️  Credibility Threshold: {trust_score_threshold}/10")
    
    print(f"\nSource Breakdown:")
    for source, articles in all_results.items():
        if articles:
            avg_credibility = sum(
                article.get('overall_credibility', {}).get('final_trust_score', 0) 
                for article in articles
            ) / len(articles)
            print(f"   {source.upper()}: {len(articles)} articles (avg: {avg_credibility:.2f}/10)")
    
    return all_results


def fetch_high_credibility_news(trust_score_threshold=7.0, max_items=30):
    """
    MOCK: Fetch only high-credibility news with minimal API usage
    EXACT same signature as real fetcher
    """
    print(f"[MOCK] Fetching High-Credibility News (threshold: {trust_score_threshold})")
    print(f"Conservative API usage: {max_items} items per source max")
    
    return fetch_all_news_enhanced(
        max_items_per_source=max_items,
        trust_score_threshold=trust_score_threshold,
        analyze_all=True
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Mock News Fetchers (Exact Match)")
    print("=" * 60)
    
    results = fetch_high_credibility_news(trust_score_threshold=6.0, max_items=10)
    
    for source, articles in results.items():
        if articles:
            print(f"\n{source.upper()}: {len(articles)} articles")
            for article in articles[:2]:
                print(f"   - {article.get('title', 'N/A')[:50]}...")
                print(f"     Score: {article.get('overall_credibility', {}).get('final_trust_score', 0):.1f}")
                print(f"     Tier: {article.get('overall_credibility', {}).get('credibility_tier', 'N/A')}")