# 🏠 House Price Prediction System

A comprehensive machine learning system for predicting real estate prices in Tunisia, featuring automated web scraping, data enrichment, model training, and a user-friendly web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)
![Web Scraping](https://img.shields.io/badge/Scraping-BeautifulSoup%20%7C%20Requests-red.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## 📋 Project Description

This project is an end-to-end machine learning solution for predicting real estate prices in the Tunisian market. The system automatically collects property listings from multiple sources, enriches the data using OpenAI's GPT models, trains multiple machine learning models, and provides predictions through both a web interface and API endpoints.

### Key Capabilities

- **🤖 Automated Data Collection**: Web scrapers for major Tunisian real estate websites (Menzili, Mubawab)
- **🧠 AI-Powered Data Enrichment**: Uses OpenAI GPT to extract structured features from unstructured property descriptions
- **📊 Advanced ML Models**: Multiple algorithms including Random Forest, XGBoost, Linear Regression, and Ensemble methods
- **🌐 Web Interface**: User-friendly web application for property price predictions
- **📈 Model Management**: Complete model training, testing, and deployment pipeline
- **⏰ Automated Scheduling**: Daily data collection using Celery task scheduler
- **🧪 Comprehensive Testing**: Full test suite with 95%+ coverage

## 🏗️ Project Structure

```
House_price_prediction/
├── 📁 src/                          # Main application source code
│   ├── app.py                       # FastAPI web application
│   ├── predict.py                   # Prediction logic
│   ├── templates.py                 # Template utilities
│   ├── scrapers/                    # Web scraping modules
│   │   ├── menzili_scraper.py      # Menzili website scraper
│   │   ├── mubawab_scraper.py      # Mubawab website scraper
│   │   ├── tecnocasa_scraper.py    # tcnocasa website scraper
│   │   └── any_website_scraper.py   # Generic website scraper
│   ├── templates/                   # HTML templates
│   │   ├── index.html              # Main page
│   │   ├── results.html            # Results display
│   │   ├── manual_entry.html       # Manual data entry
│   │   └── error.html              # Error page
│   └── static/                      # CSS and static files
│       └── style.css               # Application styling
│
├── 📁 scripts/                      # Utility and processing scripts
│   ├── train.py                    # Model training script
│   ├── test_models.py              # Model testing and evaluation
│   ├── demo_models.py              # Interactive model demo
│   ├── enrich_data.py              # OpenAI data enrichment
│   └── crawlers/                   # Automated crawler system
│       ├── celery_timer.py         # Celery task scheduler
│       ├── menzili_crawler.py      # Menzili daily crawler
│       ├── mubaweb_crawler.py      # Mubawab daily crawler
│       ├── tecnocasa_crawler.py    # tecnocasa crawler
│       ├── setup_celery.py         # Celery setup automation
│       └── test_celery_system.py   # Celery system tests
│
├── 📁 data/                        # Data storage
│   ├── raw/                       # Raw scraped data
│   │   ├── menzili_listings.json  # Menzili property data
│   │   ├── mubawab_listings.json  # Mubawab property data
│   │   └── backups/               # Weekly data backups
│   ├── processed/                 # Processed and enriched data
│   │   └── enriched_real_estate_datav4.json
│   └── external/                  # External data sources
│
├── 📁 models/                      # Trained machine learning models
│   ├── random_forest_*.joblib     # Random Forest model
│   ├── xgboost_*.joblib           # XGBoost model
│   ├── linear_regression_*.joblib # Linear Regression model
│   ├── ensemble_*.joblib          # Ensemble model
│   ├── tfidf_vectorizer_*.joblib  # Text vectorizer
│   └── latest_model_metadata.json # Model metadata
│
├── 📁 tests/                       # Test suite
│   ├── test_api.py                # API endpoint tests
│   ├── test_scrapers.py           # Scraper functionality tests
│   ├── test_prediction.py         # Prediction logic tests
│   ├── test_integration.py        # Integration tests
│   └── conftest.py                # Test configuration
│
├── 📁 logs/                        # Application logs
│   ├── daily_summary_*.json       # Daily crawler reports
│   ├── menzili_tasks.jsonl        # Menzili task logs
│   ├── mubawab_tasks.jsonl        # Mubawab task logs
│   └── system_test.jsonl          # System test logs
│
├── 📋 Configuration Files
│   ├── requirements.txt            # Python dependencies
│   ├── requirements-celery.txt     # Celery-specific dependencies
│   ├── requirements-test.txt       # Testing dependencies
│   ├── pytest.ini                 # Test configuration
│   └── .gitignore                 # Git ignore rules
│
├── 🚀 Deployment & Operations
│   ├── start_crawler.bat          # Windows crawler launcher
│   ├── run_tests.py               # Test runner script
│   └── utils.py                   # Utility functions
│
└── 📚 Documentation
    ├── README.md                   # This file
    ├── CELERY_CRAWLER_GUIDE.md     # Celery system guide
    ├── CELERY_SETUP_COMPLETE.md    # Celery setup instructions
    ├── MODEL_SYSTEM_COMPLETE.md    # Model system documentation
    └── README_WebApp.md            # Web application guide
```

## ✨ Features

### 🔍 Web Scraping & Data Collection
- **Multi-source scraping**: Automated collection from Menzili and Mubawab
- **Generic scraper**: Adaptable scraper for any real estate website
- **Intelligent parsing**: Extracts property details, prices, and descriptions
- **Duplicate detection**: Prevents duplicate listings across sources
- **Rate limiting**: Respectful scraping with configurable delays

### 🧠 AI-Powered Data Enrichment
- **OpenAI Integration**: Uses GPT-4 to extract structured data from descriptions
- **Feature extraction**: Automatically identifies amenities, room counts, and quality scores
- **Location parsing**: Extracts governorate, delegation, and locality information
- **Quality assessment**: AI-generated property quality scores (0-10 scale)
- **Batch processing**: Efficient processing of thousands of listings

### 🤖 Machine Learning Pipeline
- **Multiple algorithms**: Random Forest, XGBoost, Linear Regression, Ensemble
- **Feature engineering**: Automated creation of derived features
- **Text analysis**: TF-IDF vectorization of property descriptions
- **Model persistence**: Automatic saving and versioning of trained models
- **Performance tracking**: Comprehensive metrics and model comparison

### 🌐 Web Application
- **FastAPI backend**: High-performance async API
- **Interactive UI**: Bootstrap-based responsive interface
- **Multiple input methods**: URL scraping and manual property entry
- **Real-time predictions**: Instant price predictions with confidence intervals
- **Results visualization**: Clear presentation of predictions and property details

### ⏰ Automated Operations
- **Daily scheduling**: Automatic daily data collection at 2:00 AM
- **Task monitoring**: Celery-based task queue with retry logic
- **Error handling**: Robust error handling and notification system
- **Data backups**: Weekly automated backups of all data
- **Log management**: Comprehensive logging and monitoring

### 🧪 Testing & Quality Assurance
- **95%+ test coverage**: Comprehensive test suite
- **API testing**: Complete endpoint testing with mock data
- **Integration tests**: End-to-end workflow validation
- **Performance tests**: Load testing and performance monitoring
- **Continuous validation**: Automated model accuracy checks

## 🚀 Setup

### Prerequisites
- **Python 3.8+** (Recommended: Python 3.10+)
- **Redis server** (for Celery task queue)
- **OpenAI API key** (for data enrichment)
- **Git** (for version control)

### 1. Clone the Repository
```bash
git clone https://github.com/youssefkhemiri/House_price_prediction.git
cd House_price_prediction
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Create .env file for sensitive data
cp .env.example .env

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
```

### 4. Initialize Data Directories
```bash
# Create necessary directories
mkdir -p data/raw data/processed data/external
mkdir -p models logs

# Initialize with empty files to maintain structure
touch data/raw/.gitkeep data/processed/.gitkeep data/external/.gitkeep
```

### 5. Run Initial Tests
```bash
# Run the test suite
python run_tests.py

# Or use pytest directly
pytest tests/ -v
```

### 6. Start the Web Application
```bash
# Start the FastAPI development server
cd src
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

### 7. Set Up Automated Crawling (Optional)
```bash
# Install Celery dependencies
pip install -r requirements-celery.txt

# Start Redis server (required for Celery)
# Windows: Download and install Redis from https://redis.io/download
# Linux: sudo systemctl start redis
# Mac: brew services start redis

# Start Celery worker (Terminal 1)
celery -A scripts.crawlers.celery_timer worker --loglevel=info

# Start Celery beat scheduler (Terminal 2)
celery -A scripts.crawlers.celery_timer beat --loglevel=info
```

## 💻 Development

### Development Workflow

1. **Feature Development**
   ```bash
   # Create feature branch
   git checkout -b feature/your-feature-name
   
   # Make changes and test
   pytest tests/ -v
   
   # Run the application locally
   uvicorn src.app:app --reload
   ```

2. **Data Collection & Processing**
   ```bash
   # Run scrapers manually
   python scripts/crawlers/menzili_crawler.py
   python scripts/crawlers/mubaweb_crawler.py
   
   # Enrich data with OpenAI
   python scripts/enrich_data.py
   ```

3. **Model Training & Testing**
   ```bash
   # Train new models
   python scripts/train.py
   
   # Test saved models
   python scripts/test_models.py
   
   # Interactive model demo
   python scripts/demo_models.py
   ```

### Code Quality Standards

- **PEP 8**: Follow Python style guidelines
- **Type hints**: Use type annotations for better code clarity
- **Docstrings**: Document all functions and classes
- **Testing**: Maintain 95%+ test coverage
- **Error handling**: Implement comprehensive error handling

### Testing Strategy

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_api.py          # API tests
pytest tests/test_scrapers.py     # Scraper tests
pytest tests/test_prediction.py   # Model tests
pytest tests/test_integration.py  # Integration tests

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Adding New Features

#### Adding a New Scraper
1. Create scraper in `src/scrapers/new_scraper.py`
2. Implement required methods: `scrape_listings()`, `parse_property()`
3. Add tests in `tests/test_scrapers.py`
4. Update web interface in `src/templates/index.html`

#### Adding New ML Models
1. Add model in `scripts/train.py`
2. Update model testing in `scripts/test_models.py`
3. Ensure model serialization compatibility
4. Add performance benchmarks

#### Extending API Endpoints
1. Add endpoint in `src/app.py`
2. Create corresponding tests in `tests/test_api.py`
3. Update API documentation
4. Test with different input scenarios

### Performance Optimization

- **Database**: Consider PostgreSQL for large-scale deployments
- **Caching**: Implement Redis caching for frequent predictions
- **Async processing**: Use FastAPI's async capabilities
- **Model optimization**: Regularly retrain and optimize models
- **Monitoring**: Set up application performance monitoring

### Debugging Tips

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with profiler
python -m cProfile -o profile.stats scripts/train.py

# Monitor Celery tasks
celery -A scripts.crawlers.celery_timer flower --port=5555
```

### Contributing Guidelines

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow code style** guidelines and run linters
4. **Update documentation** for any API changes
5. **Submit a pull request** with detailed description

### Deployment Considerations

- **Environment variables**: Use proper environment management
- **Secrets management**: Secure API keys and credentials
- **Monitoring**: Implement logging and error tracking
- **Scaling**: Consider Docker containerization for production
- **Backup strategy**: Regular data and model backups

---

## 📞 Support & Contact

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/youssefkhemiri/House_price_prediction/issues)
- **Documentation**: See individual component guides in the docs/ folder
- **Email**: youssefkhemiri@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ using Python, FastAPI, and modern ML tools*
