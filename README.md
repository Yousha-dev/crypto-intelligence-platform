# 🚀 Cryptocurrency Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Django](https://img.shields.io/badge/Django-5.0+-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)]()

An advanced AI-powered platform for cryptocurrency market intelligence that combines real-time market data, news aggregation, social sentiment analysis, and intelligent Q&A capabilities through a Retrieval-Augmented Generation (RAG) system.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [API Examples](#api-examples)
- [Performance Metrics](#performance-metrics)
- [Team](#team)

## 🎯 Overview

The cryptocurrency market is characterized by extreme volatility and information overload from thousands of sources. This platform addresses these challenges by:

- **Aggregating** real-time and historical market data from multiple exchanges
- **Analyzing** sentiment and credibility of news and social media content
- **Providing** AI-powered insights through a sophisticated RAG system
- **Mapping** relationships between crypto entities in a semantic knowledge graph

### Problem Statement

- High prevalence of fake news and market manipulation
- Lack of unified systems combining price data, news, and sentiment analysis
- No source-level trust scoring with automated moderation
- Existing AI assistants prone to hallucination or rely on low-quality data

## ✨ Key Features

### 1. **Multi-Source Data Ingestion**
- Real-time OHLCV data from major exchanges via CCXT/CCXTPro
- News aggregation from CryptoPanic, CoinDesk, CryptoCompare, NewsAPI, Messari
- Social media monitoring from Reddit, Twitter, and YouTube
- WebSocket streaming with automatic reconnection and gap detection

### 2. **Advanced NLP Pipeline**
- **Sentiment Analysis**: FinBERT ensemble model (F1 score: 0.86)
- **Entity Recognition**: spaCy NER for crypto-specific entities
- **Topic Modeling**: BERTopic for trend detection (coherence: 0.68)
- **Credibility Scoring**: Weighted algorithm with ROC-AUC of 0.91

### 3. **Intelligent RAG System**
- Semantic search using FAISS vector database
- Multi-LLM support: OpenAI GPT-4, Anthropic Claude, Groq/Llama, Ollama
- Automatic failover between LLM providers
- Cross-encoder re-ranking for improved accuracy
- Knowledge graph augmentation for contextual responses

### 4. **Semantic Knowledge Graph**
- 5,000+ crypto entities mapped (coins, exchanges, people, regulators)
- Relationship reasoning and path-finding
- Dynamic updates based on market events
- Entity types: Cryptocurrencies (BTC, ETH), Exchanges (Binance, Coinbase), People (CZ, Vitalik), Regulators (SEC, CFTC)

### 5. **Credibility Scoring Engine**
Evaluates content based on:
- Source reliability and historical accuracy
- Content quality and cross-reference verification
- Recency and update frequency
- Automated moderation tiers (auto-approve, review, flag)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│         Real-time Dashboard & Visualization                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│              API Layer (Django REST + WebSocket)            │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌────▼─────┐ ┌───────▼────────┐
│  Data Ingestion│ │NLP/AI    │ │  RAG System    │
│    Pipeline    │ │Processing│ │  + Knowledge   │
│                │ │          │ │     Graph      │
│ • CCXT/Pro     │ │• FinBERT │ │• LlamaIndex    │
│ • News APIs    │ │• spaCy   │ │• Multi-LLM     │
│ • Social APIs  │ │• BERTopic│ │• FAISS Search  │
└───────┬────────┘ └────┬─────┘ └───────┬────────┘
        │                │                │
┌───────┴────────────────┴────────────────┴────────┐
│              Database Layer                      │
│                                                   │
│ • PostgreSQL (Relational)                        │
│ • InfluxDB (Time-Series)                         │
│ • MongoDB (Semi-Structured)                      │
│ • FAISS (Vector Embeddings)                      │
│ • Redis (Cache + Queue)                          │
└──────────────────────────────────────────────────┘
```

### Data Flow

1. **Market Data**: Historical backfill + WebSocket streaming → InfluxDB
2. **News/Social**: API collection → MongoDB → Credibility scoring → NLP processing
3. **RAG Pipeline**: Document embedding → FAISS indexing → Retrieval + LLM generation
4. **API Layer**: REST and WebSocket endpoints serve processed data to clients

## 🛠️ Technology Stack

### Backend
- **Framework**: Django 4.0+, Django REST Framework, Django Channels
- **Task Queue**: Celery with Redis broker
- **WebSockets**: Django Channels for real-time communication

### Databases
- **PostgreSQL**: User data and metadata
- **InfluxDB**: Time-series market data
- **MongoDB**: News and social media content
- **FAISS**: Vector embeddings for semantic search
- **Redis**: Caching, queuing, and WebSocket messaging

### AI/ML
- **NLP Models**: 
  - FinBERT (sentiment analysis)
  - spaCy (named entity recognition)
  - BERTopic (topic modeling)
  - BGE, MiniLM (embeddings)
- **RAG Framework**: LlamaIndex
- **LLM Providers**: OpenAI GPT-4, Anthropic Claude, Groq/Llama, Ollama

### Data Sources
- **Exchanges**: CCXT, CCXTPro
- **News**: CryptoPanic, CoinDesk, CryptoCompare, NewsAPI, Messari
- **Social**: Reddit API, Twitter API, YouTube Data API

### Frontend
- React with real-time visualizations
- WebSocket integration for live updates

## 🔄 How It Works

The platform operates through four integrated layers that work together to provide comprehensive cryptocurrency intelligence:

### 1. Data Collection Layer

**Market Data Ingestion**
- Connects to multiple cryptocurrency exchanges using CCXT and CCXTPro libraries
- Collects real-time price data (OHLCV - Open, High, Low, Close, Volume) via WebSocket connections
- Maintains historical data through scheduled backfill processes
- Automatically detects and fills data gaps to ensure continuity
- Stores time-series data in InfluxDB for efficient querying and analysis

**News and Social Media Aggregation**
- Continuously monitors 5 major news sources: CryptoPanic, CoinDesk, CryptoCompare, NewsAPI, and Messari
- Tracks social sentiment from Reddit, Twitter, and YouTube
- Uses scheduled tasks (Celery) to fetch new content every few minutes
- Filters duplicate content and normalizes data formats
- Stores semi-structured content in MongoDB for flexible querying

### 2. AI Processing Layer

**Natural Language Processing Pipeline**
The system processes every piece of text content through multiple NLP stages:

1. **Sentiment Analysis**: Uses FinBERT (a finance-specialized BERT model) to determine if content is positive, negative, or neutral towards specific cryptocurrencies
2. **Entity Recognition**: Employs spaCy NER to identify and extract mentions of cryptocurrencies, exchanges, people, and organizations
3. **Topic Modeling**: Applies BERTopic to cluster related content and identify trending topics in real-time
4. **Trending Detection**: Analyzes frequency and velocity of mentions to spot emerging trends

**Credibility Scoring Engine**
Each piece of content receives a credibility score (0-1) based on:
- **Source Reliability** (35%): Historical accuracy and reputation of the source
- **Content Quality** (25%): Writing quality, presence of citations, depth of analysis
- **Cross-Reference Verification** (20%): Whether claims are corroborated by other sources
- **Source History** (15%): Track record of the author or publication
- **Recency** (5%): How current the information is

Content is automatically categorized into moderation tiers:
- **Auto-Approve** (score > 0.7): High-quality content displayed immediately
- **Review Queue** (0.4-0.7): Flagged for manual review
- **Auto-Flag** (< 0.4): Potentially misleading content marked with warnings

### 3. Knowledge Graph Layer

**Semantic Entity Mapping**
- Maintains a graph database with 5,000+ entities and their relationships
- Entity types include: Cryptocurrencies (BTC, ETH), Exchanges (Binance, Coinbase), People (CZ, Vitalik Buterin), Regulators (SEC, CFTC)
- Relationships tracked: "founded_by", "regulated_by", "competes_with", "invested_in", etc.
- Automatically updates based on news events and market developments

**How It Enhances Responses**
When a user asks about an entity, the graph provides:
- Direct connections (e.g., "Who founded Binance?")
- Multi-hop reasoning (e.g., "What regulatory issues affect exchanges where Bitcoin is traded?")
- Contextual information for better AI responses

### 4. RAG (Retrieval-Augmented Generation) System

**Query Processing Workflow**

1. **Query Analysis**
   - User submits a question (e.g., "What are the recent regulatory developments affecting Bitcoin?")
   - System identifies key entities (Bitcoin, regulations) and intent

2. **Document Retrieval**
   - Converts query into a vector embedding using BGE or MiniLM models
   - Searches FAISS vector database for semantically similar content
   - Retrieves top 20-50 relevant documents from news, social media, and historical data
   - Applies credibility filter (e.g., only sources with score > 0.6)

3. **Re-Ranking**
   - Uses a cross-encoder model to re-score document relevance
   - Prioritizes more recent and higher-credibility sources
   - Selects top 5-10 documents for context

4. **Knowledge Graph Augmentation**
   - Queries the knowledge graph for entity relationships
   - Adds contextual information about mentioned entities
   - Includes recent events and connections

5. **LLM Generation**
   - Sends retrieved documents + knowledge graph context + user query to LLM
   - Primary: OpenAI GPT-4
   - Fallback chain: Anthropic Claude → Groq/Llama → Ollama (local)
   - LLM generates comprehensive answer grounded in retrieved documents
   - Includes citations with credibility scores for each source

6. **Response Delivery**
   - Returns formatted answer with inline citations
   - Provides credibility indicators for each source
   - Offers links to original articles for verification
   - Suggests related questions or entities to explore

**Multi-LLM Failover Strategy**
- If primary LLM (GPT-4) is unavailable or rate-limited, automatically switches to Claude
- If Claude fails, tries Groq/Llama for faster inference
- Final fallback to local Ollama instance
- Ensures 99%+ query success rate

### 5. Real-Time Streaming

**WebSocket Architecture**
- Maintains persistent connections to exchanges for live price updates
- Broadcasts price changes to connected clients in real-time
- Pushes breaking news and high-impact social mentions as they arrive
- Supports multiple concurrent connections with Redis pub/sub
- Automatic reconnection with exponential backoff on connection loss

### Data Flow Summary

```
User Query → RAG System
    ↓
Semantic Search → Retrieve relevant documents
    ↓
Credibility Filter → Only high-quality sources
    ↓
Knowledge Graph → Add entity context
    ↓
LLM Generation → Comprehensive answer with citations
    ↓
User receives verifiable, credibility-scored response
```

## 💡 Example Use Cases

**1. Regulatory Impact Analysis**
- User asks: "How will the SEC's new crypto regulations affect Ethereum?"
- System retrieves: Recent SEC announcements, expert analysis, historical regulatory impacts
- Knowledge graph adds: SEC's history with crypto, Ethereum Foundation relationships, past enforcement actions
- LLM generates: Comprehensive analysis with citations to official sources and expert commentary

**2. Sentiment-Driven Trading Insights**
- User asks: "What's the current sentiment around Bitcoin?"
- System aggregates: Real-time sentiment from Twitter, Reddit, news articles
- Credibility filter: Excludes low-quality sources and potential pump-and-dump schemes
- Output: Weighted sentiment score with breakdown by source type and credibility tier

**3. Entity Relationship Exploration**
- User asks: "Show me all exchanges connected to Binance's founder"
- Knowledge graph: Maps CZ → Founded Binance → Binance.US → Regulated by CFTC
- Output: Visual graph with relationship types and recent relevant news

## 🎯 API Examples

#### Get Market Data

```bash
curl -X GET "http://localhost:8000/api/v1/market/ohlcv/?symbol=BTC/USDT&timeframe=1h&limit=100"
```

#### Search News with Credibility

```bash
curl -X GET "http://localhost:8000/api/v1/news/search/?q=bitcoin&min_credibility=0.7"
```

#### RAG Query

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the recent regulatory developments affecting Bitcoin?",
    "include_sources": true,
    "min_credibility": 0.6
  }'
```

#### WebSocket Connection (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/market/BTC-USDT/');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Price update:', data);
};
```

## 📚 REST API Documentation

The platform provides a comprehensive REST API with over 100 endpoints organized into logical categories:

### Authentication & User Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register/` | POST | User registration |
| `/auth/login/` | POST | User authentication |
| `/auth/api/token/` | POST | Obtain JWT token |
| `/auth/request-password-reset/` | POST | Request password reset |
| `/auth/reset-password/` | POST | Reset user password |
| `/auth/users/change-password/` | PUT | Change user password |
| `/auth/users/deactivate-user/` | POST | Deactivate user account |
| `/auth/users/delete-user/` | DELETE | Delete user account |

### Subscription Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/subscriptionplans/` | GET | List available subscription plans |
| `/core/subscription/` | GET | Get current user subscription |
| `/core/subscription/plans/` | GET | Get subscription plan details |
| `/core/subscription/change/` | POST | Change subscription plan |
| `/core/subscription/stats/` | GET | Subscription usage statistics |
| `/core/subscription/limits/` | GET | Current plan limits |
| `/core/subscription/health/` | GET | Subscription health status |

### Payment Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/payment/create-intent/` | POST | Create payment intent (Stripe) |
| `/auth/payment/status/` | GET | Check payment status |
| `/core/payments/` | GET | List user payments |
| `/core/payments/billing/history/` | GET | Payment history |

### News & Content Feed

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/trading/content/feed/` | GET | Main content feed (news + social) |
| `/core/trading/content/articles/` | GET | News articles with filters |
| `/core/trading/content/articles/{article_id}/` | GET | Single article details |
| `/core/trading/content/search/` | GET | Search news content |
| `/core/trading/content/combined/` | GET | Combined news and social feed |

### Social Media Intelligence

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/trading/content/social/` | GET | Social media posts feed |
| `/core/trading/content/social/{post_id}/` | GET | Single post details |
| `/core/trading/content/social-search/` | GET | Search social content |
| `/core/trading/content/social-stats/` | GET | Social media statistics |

### Credibility & Content Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/trading/content/analyze/` | POST | Analyze content credibility |
| `/core/trading/content/credibility-stats/` | GET | Credibility statistics |
| `/core/trading/content/source-history/` | GET | Source reliability history |

### Sentiment & Trending Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/trading/content/sentiment-overview/` | GET | Market sentiment overview |
| `/core/trading/content/trending/` | GET | Trending topics and entities |
| `/core/trading/content/trending-history/` | GET | Historical trending data |
| `/core/trading/content/topics/` | GET | Topic modeling results |

### Market Insights

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/trading/insights/summary/` | GET | Market insights summary |
| `/core/trading/insights/coin/{symbol}/` | GET | Specific cryptocurrency insights |
| `/core/trading/insights/alerts/` | GET | Market alerts and notifications |
| `/core/trading/insights/narratives/` | GET | Dominant market narratives |
| `/core/trading/insights/influencers/` | GET | Top crypto influencers |
| `/core/trading/exchanges/` | GET | Supported exchanges list |

### RAG System - Q&A & Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/trading/rag/ask/` | POST | Single-turn Q&A |
| `/core/trading/rag/chat/` | POST | Multi-turn conversation |
| `/core/trading/rag/chat-context/` | POST | Chat with context |
| `/core/trading/rag/query-chain/` | POST | Multi-step query chains |

### RAG System - Analysis & Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/trading/rag/semantic-search/` | GET | Semantic search over content |
| `/core/trading/rag/analyze-entity/` | GET | Deep entity analysis |
| `/core/trading/rag/market-sentiment/` | GET | AI-powered sentiment analysis |
| `/core/trading/rag/summary/` | POST | Generate content summaries |
| `/core/trading/rag/performance/` | GET | RAG system performance metrics |

### Admin - User Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/all-users/` | GET | List all users |
| `/admin/users/` | GET | Get users with filters |
| `/admin/users/{user_id}/edit-user` | PUT | Edit user details |
| `/admin/dashboard/stats/` | GET | Admin dashboard statistics |

### Admin - Subscription Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/subscriptionplans/list/` | GET | List subscription plans |
| `/admin/subscriptionplans/create/` | POST | Create subscription plan |
| `/admin/subscriptionplans/{id}/update/` | PUT | Update subscription plan |
| `/admin/subscriptionplans/{id}/delete/` | DELETE | Delete subscription plan |
| `/admin/subscriptionplans/dashboard/` | GET | Subscription dashboard |
| `/admin/subscriptionplans/analytics/` | GET | Subscription analytics |

### Admin - Subscription Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/subscriptions/list/` | GET | List all subscriptions |
| `/admin/subscriptions/create/` | POST | Create subscription |
| `/admin/subscriptions/{id}/update/` | PUT | Update subscription |
| `/admin/subscriptions/{id}/delete/` | DELETE | Delete subscription |
| `/admin/subscriptions/{id}/renew/` | POST | Manually renew subscription |
| `/admin/subscriptions/sync/auto-renew/` | POST | Sync auto-renewal |
| `/admin/subscriptions/dashboard/` | GET | Subscriptions dashboard |
| `/admin/subscriptions/analytics/` | GET | Subscription analytics |

### Admin - Content Moderation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/content/moderation/queue/` | GET | Content moderation queue |
| `/admin/content/moderation/approve/{article_id}/` | POST | Approve content |
| `/admin/content/moderation/reject/{article_id}/` | POST | Reject content |
| `/admin/content/moderation/flag/{article_id}/` | POST | Flag content for review |

### Admin - System Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/content/health/` | GET | Content system health |
| `/admin/content/statistics/` | GET | Content statistics |
| `/admin/content/thresholds/` | POST | Set content thresholds |
| `/admin/payments/` | GET | All payments |

### Admin - RAG Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/rag/rag-stats/` | GET | RAG system statistics |
| `/admin/rag/index-documents/` | POST | Index new documents |
| `/admin/rag/rebuild-index/` | POST | Rebuild vector index |

### Admin - Knowledge Graph

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/rag/kg/build/` | POST | Build/update knowledge graph |
| `/admin/rag/kg/stats/` | GET | Knowledge graph statistics |
| `/admin/rag/kg/entity/` | GET | Query entities |
| `/admin/rag/kg/path/` | GET | Find entity paths |
| `/admin/rag/kg/trending/` | GET | Trending entities |

### Admin - LLM Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/rag/llm/status/` | GET | LLM provider status |
| `/admin/rag/llm/switch/` | POST | Switch LLM provider |
| `/admin/rag/llm/test/` | POST | Test LLM connection |

### User Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/core/user/` | GET | Get current user profile |
| `/core/user/update/` | PUT | Update user profile |
| `/core/usage/record/` | POST | Record API usage |

### WebSocket Endpoints

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/ws/market/{symbol}/` | WebSocket | Real-time price updates |
| `/ws/news/` | WebSocket | Real-time news feed |
| `/ws/social/` | WebSocket | Real-time social mentions |
| `/ws/system/status/` | WebSocket | System health monitoring |

## 📊 Performance Metrics

### System Performance
- **Throughput**: 1,000+ articles processed daily
- **Latency**: <5-minute processing time
- **Uptime**: >90% for live market data streaming
- **Scalability**: Horizontal scaling via Celery task queues

### AI Model Performance
- **Sentiment Analysis F1**: 0.86
- **Credibility Detection ROC-AUC**: 0.91
- **Topic Modeling Coherence**: 0.68
- **RAG Response Time**: <2s for complex queries
- **FAISS Query Latency**: <50ms
- **Redis Cache Retrieval**: <100ms

### Data Processing
- **ETL Pipeline Latency**: <300ms per unit
- **Event Processing**: 1,000+ events/hour sustained
- **WebSocket Reconnection**: Automatic with gap detection

## 👥 Team

**Project Supervisor**
- Mr. Zulfiqar Memon

**Development Team**
- **Yousha Masood** (K21-3928) - [GitHub](https://github.com/Yousha-dev) | [LinkedIn](https://linkedin.com/in/yousha-masood)
- **Yahya Hussain** (K21-4895) - [GitHub](https://github.com/yahya-hussain) | [LinkedIn](https://linkedin.com/in/yahya-hussain)
- **Bilal Farooqui** (K21-3920) - [GitHub](https://github.com/bilal-farooqui) | [LinkedIn](https://linkedin.com/in/bilal-farooqui)

**Institution**
- FAST School of Computing
- National University of Computer and Emerging Sciences (NUCES)
- Karachi Campus

## 🙏 Acknowledgments

- FinBERT team for domain-specific NLP models
- LlamaIndex for RAG framework
- CCXT for exchange integration
- Anthropic, OpenAI, and Groq for LLM access
- All open-source contributors whose libraries made this possible

---

**Made with ❤️**

*Submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Computer Science, December 2025*