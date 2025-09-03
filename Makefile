# House Price Prediction - Makefile
# Automated ML Pipeline for Real Estate Price Prediction
#
# This Makefile orchestrates the complete machine learning pipeline:
# 1. Data Collection (Web Scraping)
# 2. Data Combination (Multiple Sources)
# 3. Data Enrichment (OpenAI Processing)  
# 4. Model Training (Multiple Algorithms)
# 5. Model Testing and Validation
#
# Usage:
#   make all          - Run complete pipeline
#   make data         - Data collection and processing only
#   make models       - Model training only
#   make test         - Run test suite
#   make clean        - Clean generated files
#   make help         - Show available targets

# =============================================================================
# Configuration Variables
# =============================================================================

# Python interpreter
PYTHON := python

# Directory paths
DATA_RAW_DIR := data/raw
DATA_PROCESSED_DIR := data/processed
MODELS_DIR := models
LOGS_DIR := logs
SCRIPTS_DIR := scripts

# Data files
COMBINED_DATA := $(DATA_RAW_DIR)/combined_data.json
ENRICHED_DATA := $(DATA_PROCESSED_DIR)/enriched_real_estate_data.json
MENZILI_DATA := $(DATA_RAW_DIR)/menzili_listings.json
MUBAWAB_DATA := $(DATA_RAW_DIR)/mubawab_listings.json

# Model files
LATEST_MODEL_METADATA := $(MODELS_DIR)/latest_model_metadata.json

# Script files
COMBINE_SCRIPT := $(SCRIPTS_DIR)/combine_data.py
ENRICH_SCRIPT := $(SCRIPTS_DIR)/enrich_data.py
TRAIN_SCRIPT := $(SCRIPTS_DIR)/train.py
TEST_SCRIPT := $(SCRIPTS_DIR)/test_models.py
DEMO_SCRIPT := $(SCRIPTS_DIR)/demo_models.py

# OpenAI and processing parameters
OPENAI_MODEL := gpt-4o
ENRICH_LIMIT := 2000
MAX_FEATURES := 100

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# =============================================================================
# Main Targets
# =============================================================================

.PHONY: all data models test clean help setup check-env

# Default target
all: check-env setup data models test-models
	@echo "$(GREEN)🎉 Complete ML pipeline finished successfully!$(RESET)"
	@echo "$(CYAN)📊 Run 'make demo' to test your models interactively$(RESET)"

# Data pipeline only
data: check-env setup combine-data enrich-data
	@echo "$(GREEN)📊 Data pipeline completed successfully!$(RESET)"

# Model training only  
models: check-env $(ENRICHED_DATA) train-models
	@echo "$(GREEN)🤖 Model training completed successfully!$(RESET)"

# Complete testing suite
test: test-api test-scrapers test-models test-integration
	@echo "$(GREEN)✅ All tests completed successfully!$(RESET)"

# =============================================================================
# Setup and Environment
# =============================================================================

setup:
	@echo "$(BLUE)🔧 Setting up directories and environment...$(RESET)"
	@mkdir -p $(DATA_RAW_DIR) $(DATA_PROCESSED_DIR) $(MODELS_DIR) $(LOGS_DIR)
	@touch $(DATA_RAW_DIR)/.gitkeep $(DATA_PROCESSED_DIR)/.gitkeep
	@echo "$(GREEN)✅ Setup completed$(RESET)"

check-env:
	@echo "$(BLUE)🔍 Checking environment...$(RESET)"
	@$(PYTHON) --version || (echo "$(RED)❌ Python not found$(RESET)" && exit 1)
	@$(PYTHON) -c "import pandas, sklearn, xgboost, openai" 2>/dev/null || \
		(echo "$(RED)❌ Missing required packages. Run: pip install -r requirements.txt$(RESET)" && exit 1)
	@test -f .env || (echo "$(YELLOW)⚠️  .env file not found. OpenAI features may not work.$(RESET)")
	@echo "$(GREEN)✅ Environment check passed$(RESET)"

install-deps:
	@echo "$(BLUE)📦 Installing Python dependencies...$(RESET)"
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(RESET)"

# =============================================================================
# Data Collection and Processing
# =============================================================================

# Combine raw data from multiple sources
combine-data: $(COMBINED_DATA)

$(COMBINED_DATA): $(MENZILI_DATA) $(MUBAWAB_DATA) $(COMBINE_SCRIPT)
	@echo "$(BLUE)🔄 Combining raw data from multiple sources...$(RESET)"
	$(PYTHON) $(COMBINE_SCRIPT) \
		--raw_dir $(DATA_RAW_DIR) \
		--output $(COMBINED_DATA) \
		--validate
	@echo "$(GREEN)✅ Data combination completed: $(COMBINED_DATA)$(RESET)"

# Enrich data with OpenAI
enrich-data: $(ENRICHED_DATA)

$(ENRICHED_DATA): $(COMBINED_DATA) $(ENRICH_SCRIPT)
	@echo "$(BLUE)🤖 Enriching data with OpenAI ($(OPENAI_MODEL))...$(RESET)"
	@echo "$(YELLOW)⏳ This may take a while for large datasets...$(RESET)"
	$(PYTHON) $(ENRICH_SCRIPT) \
		--input $(COMBINED_DATA) \
		--output $(ENRICHED_DATA) \
		--model $(OPENAI_MODEL) \
		--limit $(ENRICH_LIMIT)
	@echo "$(GREEN)✅ Data enrichment completed: $(ENRICHED_DATA)$(RESET)"

# Manual data collection targets
collect-menzili:
	@echo "$(BLUE)🏠 Collecting data from Menzili...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/crawlers/menzili_crawler.py
	@echo "$(GREEN)✅ Menzili data collection completed$(RESET)"

collect-mubawab:
	@echo "$(BLUE)🏠 Collecting data from Mubawab...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/crawlers/mubaweb_crawler.py  
	@echo "$(GREEN)✅ Mubawab data collection completed$(RESET)"

collect-all: collect-menzili collect-mubawab
	@echo "$(GREEN)🎉 All data collection completed$(RESET)"

# =============================================================================
# Model Training and Management  
# =============================================================================

# Train all models
train-models: $(LATEST_MODEL_METADATA)

$(LATEST_MODEL_METADATA): $(ENRICHED_DATA) $(TRAIN_SCRIPT)
	@echo "$(BLUE)🤖 Training machine learning models...$(RESET)"
	@echo "$(YELLOW)⏳ This will train multiple algorithms and may take several minutes...$(RESET)"
	$(PYTHON) $(TRAIN_SCRIPT) \
		--data_path $(ENRICHED_DATA) \
		--model_dir $(MODELS_DIR)
	@echo "$(GREEN)✅ Model training completed$(RESET)"

# Train models without TF-IDF features
train-no-tfidf: $(ENRICHED_DATA)
	@echo "$(BLUE)🤖 Training models without TF-IDF features...$(RESET)"
	$(PYTHON) $(TRAIN_SCRIPT) \
		--data_path $(ENRICHED_DATA) \
		--model_dir $(MODELS_DIR) \
		--no_tfidf

# Train models with TF-IDF features only
train-tfidf-only: $(ENRICHED_DATA)
	@echo "$(BLUE)🤖 Training models with TF-IDF features only...$(RESET)"
	$(PYTHON) $(TRAIN_SCRIPT) \
		--data_path $(ENRICHED_DATA) \
		--model_dir $(MODELS_DIR) \
		--tfidf_only

# Quick training for testing (no model saving)
train-test: $(ENRICHED_DATA)
	@echo "$(BLUE)🧪 Quick training for testing (models not saved)...$(RESET)"
	$(PYTHON) $(TRAIN_SCRIPT) \
		--data_path $(ENRICHED_DATA) \
		--no_save

# =============================================================================
# Model Testing and Validation
# =============================================================================

# Test saved models
test-models: $(LATEST_MODEL_METADATA)
	@echo "$(BLUE)🧪 Testing saved models...$(RESET)"
	$(PYTHON) $(TEST_SCRIPT)
	@echo "$(GREEN)✅ Model testing completed$(RESET)"

# Test models on specific data
test-models-data: $(LATEST_MODEL_METADATA)
	@echo "$(BLUE)🧪 Testing models on validation data...$(RESET)"
	@if [ -f "$(DATA_PROCESSED_DIR)/validation_data.json" ]; then \
		$(PYTHON) $(TEST_SCRIPT) --data_path $(DATA_PROCESSED_DIR)/validation_data.json; \
	else \
		echo "$(YELLOW)⚠️  No validation data found. Using training data for demo.$(RESET)"; \
		$(PYTHON) $(TEST_SCRIPT) --data_path $(ENRICHED_DATA); \
	fi

# Interactive model demonstration
demo: $(LATEST_MODEL_METADATA)
	@echo "$(BLUE)🎮 Starting interactive model demo...$(RESET)"
	$(PYTHON) $(DEMO_SCRIPT)

# Single prediction example
predict-sample: $(LATEST_MODEL_METADATA)
	@echo "$(BLUE)🔮 Making sample prediction...$(RESET)"
	$(PYTHON) $(TEST_SCRIPT) --single_prediction \
		'{"room_count": 3, "land_area": 150, "has_garage": 1, "has_garden": 1, "price": 300000}'

# =============================================================================
# Testing and Quality Assurance
# =============================================================================

# API endpoint tests
test-api:
	@echo "$(BLUE)🧪 Running API tests...$(RESET)"
	$(PYTHON) -m pytest tests/test_api.py -v
	@echo "$(GREEN)✅ API tests completed$(RESET)"

# Scraper functionality tests
test-scrapers:
	@echo "$(BLUE)🧪 Running scraper tests...$(RESET)"
	$(PYTHON) -m pytest tests/test_scrapers.py -v
	@echo "$(GREEN)✅ Scraper tests completed$(RESET)"

# Integration tests
test-integration:
	@echo "$(BLUE)🧪 Running integration tests...$(RESET)"
	$(PYTHON) -m pytest tests/test_integration.py -v
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

# Complete test suite
test-all:
	@echo "$(BLUE)🧪 Running complete test suite...$(RESET)"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html
	@echo "$(GREEN)✅ All tests completed$(RESET)"

# =============================================================================
# Web Application
# =============================================================================

# Start the web application
serve:
	@echo "$(BLUE)🌐 Starting web application...$(RESET)"
	@echo "$(CYAN)🔗 Open http://localhost:8000 in your browser$(RESET)"
	cd src && $(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Start web app in background
serve-bg:
	@echo "$(BLUE)🌐 Starting web application in background...$(RESET)"
	cd src && $(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000 &
	@echo "$(GREEN)✅ Web app started at http://localhost:8000$(RESET)"

# =============================================================================
# Automated Crawling (Celery)
# =============================================================================

# Setup Celery environment
setup-celery:
	@echo "$(BLUE)⏰ Setting up Celery environment...$(RESET)"
	$(PYTHON) -m pip install -r requirements-celery.txt
	$(PYTHON) $(SCRIPTS_DIR)/crawlers/setup_celery.py
	@echo "$(GREEN)✅ Celery setup completed$(RESET)"

# Start Celery worker (blocking)
celery-worker:
	@echo "$(BLUE)⏰ Starting Celery worker...$(RESET)"
	@echo "$(YELLOW)📝 Keep this terminal open$(RESET)"
	celery -A scripts.crawlers.celery_timer worker --loglevel=info

# Start Celery beat scheduler (blocking)
celery-beat:
	@echo "$(BLUE)⏰ Starting Celery beat scheduler...$(RESET)"  
	@echo "$(YELLOW)📝 Keep this terminal open$(RESET)"
	celery -A scripts.crawlers.celery_timer beat --loglevel=info

# Start Celery monitoring interface
celery-monitor:
	@echo "$(BLUE)📊 Starting Celery monitoring interface...$(RESET)"
	@echo "$(CYAN)🔗 Open http://localhost:5555 in your browser$(RESET)"
	celery -A scripts.crawlers.celery_timer flower --port=5555

# Test Celery system
test-celery:
	@echo "$(BLUE)🧪 Testing Celery system...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/crawlers/test_celery_system.py

# =============================================================================
# Maintenance and Cleanup
# =============================================================================

# Clean generated files
clean:
	@echo "$(BLUE)🧹 Cleaning generated files...$(RESET)"
	rm -f $(COMBINED_DATA)
	rm -f $(ENRICHED_DATA)
	rm -f before_enriched_real_estate_data.json
	rm -rf $(MODELS_DIR)/*.joblib
	rm -rf $(MODELS_DIR)/*.json
	rm -rf $(LOGS_DIR)/*
	rm -rf __pycache__ src/__pycache__ scripts/__pycache__
	rm -rf .pytest_cache htmlcov
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

# Clean only data files (keep models)
clean-data:
	@echo "$(BLUE)🧹 Cleaning data files...$(RESET)"
	rm -f $(COMBINED_DATA)
	rm -f $(ENRICHED_DATA)
	rm -f before_enriched_real_estate_data.json
	@echo "$(GREEN)✅ Data cleanup completed$(RESET)"

# Clean only models
clean-models:
	@echo "$(BLUE)🧹 Cleaning model files...$(RESET)"
	rm -rf $(MODELS_DIR)/*.joblib
	rm -rf $(MODELS_DIR)/*.json
	@echo "$(GREEN)✅ Model cleanup completed$(RESET)"

# Backup important files
backup:
	@echo "$(BLUE)💾 Creating backup...$(RESET)"
	@BACKUP_DIR=backup_$(shell date +%Y%m%d_%H%M%S) && \
	mkdir -p $$BACKUP_DIR && \
	cp -r $(DATA_RAW_DIR) $$BACKUP_DIR/ && \
	cp -r $(DATA_PROCESSED_DIR) $$BACKUP_DIR/ && \
	cp -r $(MODELS_DIR) $$BACKUP_DIR/ && \
	echo "$(GREEN)✅ Backup created: $$BACKUP_DIR$(RESET)"

# =============================================================================
# Information and Statistics
# =============================================================================

# Show pipeline status
status:
	@echo "$(BLUE)📊 Pipeline Status$(RESET)"
	@echo "$(WHITE)==================$(RESET)"
	@echo -n "Raw Data (Combined): "
	@if [ -f $(COMBINED_DATA) ]; then \
		echo "$(GREEN)✅ Available ($(shell stat -f%z $(COMBINED_DATA) 2>/dev/null || stat -c%s $(COMBINED_DATA) 2>/dev/null || echo "Unknown") bytes)$(RESET)"; \
	else \
		echo "$(RED)❌ Missing$(RESET)"; \
	fi
	@echo -n "Enriched Data: "
	@if [ -f $(ENRICHED_DATA) ]; then \
		echo "$(GREEN)✅ Available ($(shell wc -l < $(ENRICHED_DATA) 2>/dev/null || echo "Unknown") records)$(RESET)"; \
	else \
		echo "$(RED)❌ Missing$(RESET)"; \
	fi
	@echo -n "Trained Models: "
	@if [ -f $(LATEST_MODEL_METADATA) ]; then \
		echo "$(GREEN)✅ Available ($(shell ls $(MODELS_DIR)/*.joblib 2>/dev/null | wc -l || echo "0") models)$(RESET)"; \
	else \
		echo "$(RED)❌ Missing$(RESET)"; \
	fi

# Show data statistics
stats: $(ENRICHED_DATA)
	@echo "$(BLUE)📊 Data Statistics$(RESET)"
	@echo "$(WHITE)==================$(RESET)"
	$(PYTHON) -c "
import pandas as pd
import json
try:
    with open('$(ENRICHED_DATA)', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(data)
    print(f'Total records: {len(df):,}')
    if 'price' in df.columns:
        print(f'Price range: {df[\"price\"].min():,.0f} - {df[\"price\"].max():,.0f} TND')
        print(f'Average price: {df[\"price\"].mean():,.0f} TND')
    if 'data_source' in df.columns:
        print('Sources:', df['data_source'].value_counts().to_dict())
except Exception as e:
    print(f'Error reading data: {e}')
	"

# Show model performance
model-performance: $(LATEST_MODEL_METADATA)
	@echo "$(BLUE)📊 Model Performance$(RESET)"
	@echo "$(WHITE)=====================$(RESET)"
	$(PYTHON) -c "
import json
try:
    with open('$(LATEST_MODEL_METADATA)', 'r') as f:
        metadata = json.load(f)
    results = metadata['training_metadata']['results']
    for model, metrics in results.items():
        r2 = metrics.get('r2_score', 0)
        mae = metrics.get('mae', 0)
        print(f'{model:20}: R² = {r2:.3f}, MAE = {mae:8,.0f} TND')
except Exception as e:
    print(f'Error reading model metadata: {e}')
	"

# =============================================================================
# Development and Debugging
# =============================================================================

# Development mode - quick pipeline for testing
dev: check-env setup
	@echo "$(BLUE)🚀 Development mode - Quick pipeline$(RESET)"
	@$(MAKE) combine-data
	$(PYTHON) $(ENRICH_SCRIPT) --input $(COMBINED_DATA) --output $(ENRICHED_DATA) --limit 100
	$(PYTHON) $(TRAIN_SCRIPT) --data_path $(ENRICHED_DATA) --no_save
	@echo "$(GREEN)✅ Development pipeline completed$(RESET)"

# Debug mode with verbose output
debug:
	@echo "$(BLUE)🐛 Debug mode$(RESET)"
	@$(MAKE) status
	@$(MAKE) stats || echo "$(YELLOW)⚠️  No enriched data available$(RESET)"
	@$(MAKE) model-performance || echo "$(YELLOW)⚠️  No trained models available$(RESET)"

# Profile the training process
profile: $(ENRICHED_DATA)
	@echo "$(BLUE)⚡ Profiling training process...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats $(TRAIN_SCRIPT) --data_path $(ENRICHED_DATA) --no_save
	@echo "$(GREEN)✅ Profile saved to profile.stats$(RESET)"

# =============================================================================
# Help and Documentation
# =============================================================================

help:
	@echo "$(BLUE)🏠 House Price Prediction - Makefile Help$(RESET)"
	@echo "$(WHITE)============================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Main Targets:$(RESET)"
	@echo "  $(GREEN)make all$(RESET)           - Run complete ML pipeline"
	@echo "  $(GREEN)make data$(RESET)          - Data collection and processing only"
	@echo "  $(GREEN)make models$(RESET)        - Model training only"
	@echo "  $(GREEN)make test$(RESET)          - Run complete test suite"
	@echo "  $(GREEN)make serve$(RESET)         - Start web application"
	@echo "  $(GREEN)make demo$(RESET)          - Interactive model demonstration"
	@echo ""
	@echo "$(YELLOW)Data Pipeline:$(RESET)"
	@echo "  $(GREEN)make combine-data$(RESET)  - Combine raw data from multiple sources"
	@echo "  $(GREEN)make enrich-data$(RESET)   - Enrich data with OpenAI"
	@echo "  $(GREEN)make collect-all$(RESET)   - Collect data from all websites"
	@echo ""
	@echo "$(YELLOW)Model Training:$(RESET)"
	@echo "  $(GREEN)make train-models$(RESET)  - Train all models with default settings"
	@echo "  $(GREEN)make train-no-tfidf$(RESET) - Train without text features"
	@echo "  $(GREEN)make train-tfidf-only$(RESET) - Train with text features only"
	@echo ""
	@echo "$(YELLOW)Testing:$(RESET)"
	@echo "  $(GREEN)make test-models$(RESET)   - Test saved models"
	@echo "  $(GREEN)make test-api$(RESET)      - Test API endpoints"
	@echo "  $(GREEN)make test-all$(RESET)      - Run complete test suite with coverage"
	@echo ""
	@echo "$(YELLOW)Automation:$(RESET)"
	@echo "  $(GREEN)make setup-celery$(RESET)  - Setup automated crawling system"
	@echo "  $(GREEN)make celery-worker$(RESET) - Start Celery worker (keep terminal open)"
	@echo "  $(GREEN)make celery-beat$(RESET)   - Start Celery scheduler (keep terminal open)"
	@echo "  $(GREEN)make celery-monitor$(RESET) - Start monitoring interface"
	@echo ""
	@echo "$(YELLOW)Information:$(RESET)"
	@echo "  $(GREEN)make status$(RESET)        - Show pipeline status"
	@echo "  $(GREEN)make stats$(RESET)         - Show data statistics"
	@echo "  $(GREEN)make model-performance$(RESET) - Show model performance metrics"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)         - Clean all generated files"
	@echo "  $(GREEN)make clean-data$(RESET)    - Clean data files only"
	@echo "  $(GREEN)make backup$(RESET)        - Create backup of important files"
	@echo ""
	@echo "$(YELLOW)Development:$(RESET)"
	@echo "  $(GREEN)make dev$(RESET)           - Quick development pipeline"
	@echo "  $(GREEN)make debug$(RESET)         - Debug mode with detailed info"
	@echo "  $(GREEN)make profile$(RESET)       - Profile training performance"
	@echo ""
	@echo "$(CYAN)Examples:$(RESET)"
	@echo "  $(WHITE)make all$(RESET)                    # Complete pipeline"
	@echo "  $(WHITE)make data && make models$(RESET)    # Step by step"
	@echo "  $(WHITE)make clean && make all$(RESET)      # Fresh start"
	@echo "  $(WHITE)make serve$(RESET)                  # Start web interface"
	@echo ""
	@echo "$(CYAN)For more information, see README.md$(RESET)"

# Show quick usage
usage:
	@echo "$(BLUE)🚀 Quick Usage$(RESET)"
	@echo "$(WHITE)===============$(RESET)"
	@echo "1. $(GREEN)make all$(RESET)     - Run complete pipeline"
	@echo "2. $(GREEN)make serve$(RESET)   - Start web application"
	@echo "3. $(GREEN)make demo$(RESET)    - Test models interactively"

# =============================================================================
# Default and Special Targets
# =============================================================================

# Prevent make from deleting intermediate files
.PRECIOUS: $(COMBINED_DATA) $(ENRICHED_DATA) $(LATEST_MODEL_METADATA)

# Disable implicit rules to speed up make
.SUFFIXES:

# Show help by default if no target specified
.DEFAULT_GOAL := help
