# Real Estate Scraper & Price Prediction Web Application

A comprehensive FastAPI-based web application for scraping real estate property data from multiple websites and predicting property prices using machine learning.

## ✨ Features

### 🔍 Web Scraping
- **Multi-Provider Support**: Tecnocasa, Menzili, Mubawab, and any website
- **Beautiful Interface**: Responsive design with Bootstrap 5 and Font Awesome icons
- **Smart Validation**: URL validation based on selected provider
- **Comprehensive Data**: Price, location, features, photos, contact info, and more

### 🧠 AI Price Prediction
- **Machine Learning Integration**: Trained model with 60+ engineered features
- **Real-time Predictions**: Instant price estimates with confidence scores
- **Feature Engineering**: Advanced text analysis, location scoring, and amenity detection
- **Price Range Estimates**: Minimum and maximum price predictions
- **Confidence Analysis**: Reliability scoring based on data completeness

### 🎨 User Experience
- **Dual Input Methods**: Web scraping OR manual property entry
- **Mobile-Responsive**: Works perfectly on all devices
- **Interactive Results**: Photo galleries, detailed breakdowns, and export options
- **Loading Animations**: Smooth user experience with progress indicators
- **Provider-Specific Themes**: Color-coded interface for each scraper

### 📊 Data Management
- **JSON Export**: Download scraped data in structured format
- **API Documentation**: Auto-generated Swagger UI and ReDoc
- **Error Handling**: Comprehensive error messages and graceful fallbacks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- All required packages (see Dependencies section)
- Trained ML model file: `models/best_model_20250906_043926.pkl`

### Installation & Launch

1. **Navigate to the application directory**
   ```bash
   cd src
   ```

2. **Start the application**
   ```bash
   python app.py
   ```
   
   Or use the convenient batch file:
   ```bash
   start_webapp.bat
   ```

3. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - The application will be available on your local network at `http://0.0.0.0:5000`

### Using the Web Interface

#### Method 1: Web Scraping
1. **Select Provider** - Choose from Tecnocasa, Menzili, Mubawab, or Any Website
2. **Enter Property URL** - Paste the complete URL of the listing
3. **Submit & Wait** - Click "Scrape Property Data" and wait for processing
4. **View Results** - See extracted data with AI price prediction

#### Method 2: Manual Entry
1. **Click "Manual Entry"** from the main page
2. **Fill Property Details** - Enter known information about the property
3. **Submit** - Get AI price prediction based on entered data

## 🤖 AI Price Prediction

Our advanced machine learning model provides:

- **Accurate Predictions**: Based on 60+ engineered features
- **Confidence Scoring**: Reliability assessment (0-100%)
- **Price Ranges**: Minimum and maximum estimated values
- **Feature Analysis**: Key factors influencing the prediction
- **Market Comparison**: How the prediction compares to listed price

### Prediction Features Include:
- **Location Analysis**: Governorate, delegation, coastal proximity, major cities
- **Property Characteristics**: Type, size, rooms, bathrooms, condition
- **Amenities & Features**: Garage, garden, pool, luxury indicators
- **Text Analysis**: Description quality, luxury keywords, investment potential
- **Market Segmentation**: Property tier and market segment classification

## 📚 API Documentation & Endpoints

### Interactive Documentation
- **Swagger UI**: `http://localhost:5000/docs` - Interactive API testing
- **ReDoc**: `http://localhost:5000/redoc` - Clean API documentation

### Available Endpoints

#### 1. Web Scraping API
```bash
POST /api/scrape
Content-Type: application/json

{
  "provider": "tecnocasa|menzili|mubawab|any_website",
  "url": "https://website.com/property-url"
}

Response:
{
  "success": true,
  "data": { /* scraped property data */ },
  "prediction": { /* AI price prediction */ }
}
```

#### 2. Price Prediction API
```bash
POST /api/predict
Content-Type: application/json

{
  "property_data": {
    "property_type": "Villa",
    "living_area": "200",
    "room_count": 4,
    "governorate": "Tunis",
    "description": ["Modern villa with garden"],
    "has_garage": true,
    // ... other property details
  }
}

Response:
{
  "success": true,
  "prediction": {
    "predicted_price": 450000,
    "confidence": 0.85,
    "price_range": {"min": 400000, "max": 500000},
    "factors": ["location", "size", "features"]
  }
}
```

## 🌐 Supported Websites

| Provider | Website | Status | Features |
|----------|---------|--------|----------|
| **Tecnocasa** | tecnocasa.tn | ✅ Active | Full property details, photos, contact |
| **Menzili** | menzili.tn | ✅ Active | Complete data extraction |
| **Mubawab** | mubawab.tn | ✅ Active | Property specs, location, amenities |
| **Any Website** | Universal | ✅ Active | Generic property scraper |

### Website-Specific Features
- **Tecnocasa**: Red-themed interface, specialized for luxury properties
- **Menzili**: Comprehensive location data and property features
- **Mubawab**: Strong amenity detection and photo extraction
- **Any Website**: Flexible scraper for unlisted real estate sites

## 📋 Extracted Data Structure

The application extracts comprehensive property information:

### Basic Information
- **Property Type**: House, apartment, villa, terrain, etc.
- **Transaction Type**: Sale, rent, or other
- **Price & Currency**: Market price in local currency
- **Listing Details**: Date posted, last updated, property ID

### Location Data
- **Address**: Full street address when available
- **Governorate**: Primary administrative region
- **Delegation**: Secondary administrative division
- **Locality**: Specific neighborhood or area
- **Postal Code**: When available

### Property Specifications
- **Living Area**: Interior square footage/meters
- **Land Area**: Lot size for properties with land
- **Room Count**: Total number of rooms
- **Bathroom Count**: Number of bathrooms
- **Floor**: Which floor (for apartments)
- **Construction Year**: Year built

### Features & Amenities
- **Parking**: Garage, parking space availability
- **Outdoor Spaces**: Garden, terrace, balcony
- **Utilities**: Pool, elevator, air conditioning
- **Condition**: New, renovated, needs work
- **Furnishing**: Furnished or unfurnished

### Rich Media & Contact
- **Photo Gallery**: High-resolution property images
- **Description**: Detailed property description
- **Contact Info**: Phone numbers, agency details
- **Agent Information**: Contact person, agency name

## 🛠️ Technical Dependencies

### Core Framework
- **FastAPI** - High-performance web framework
- **Uvicorn** - ASGI server for FastAPI
- **Jinja2** - Template engine for HTML rendering
- **Python-multipart** - Form data handling

### Web Scraping
- **Requests** - HTTP library for web requests
- **BeautifulSoup4** - HTML parsing and extraction
- **lxml** - Fast XML/HTML parser
- **Selenium** - Browser automation (for dynamic sites)

### Machine Learning
- **scikit-learn** - ML model training and prediction
- **joblib** - Model serialization and loading
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

### UI/UX
- **Bootstrap 5.3.0** - Responsive CSS framework
- **Font Awesome 6.4.0** - Icon library
- **Custom CSS** - Enhanced styling and themes

## 🏗️ Project Structure

```
src/
├── app.py                 # Main FastAPI application
├── predict.py            # ML prediction module
├── templates/
│   ├── index.html        # Main scraper interface
│   ├── manual_entry.html # Manual property input
│   └── results.html      # Results display page
├── scrapers/
│   ├── tecnocasa_scraper.py    # Tecnocasa website scraper
│   ├── menzili_scraper.py      # Menzili website scraper
│   ├── mubawab_scraper.py      # Mubawab website scraper
│   └── any_website_scraper.py  # Universal scraper
└── static/               # CSS and static files

models/
└── best_model_20250906_043926.pkl  # Trained ML model
```

## 🔧 Development & Customization

### Adding New Website Support

1. **Create Scraper Module**
   ```python
   # scrapers/new_site_scraper.py
   def scrape_new_site(url):
       # Implementation here
       return property_data
   ```

2. **Update Main Application**
   ```python
   # app.py
   SUPPORTED_PROVIDERS['new_site'] = {
       'name': 'New Site',
       'domain': 'newsite.com',
       'description': 'Description',
       'icon': 'building'
   }
   ```

3. **Add Import and Logic**
   ```python
   from scrapers.new_site_scraper import scrape_new_site
   
   # Add to scraping logic
   elif provider == 'new_site':
       data = scrape_new_site(url)
   ```

### Customizing the ML Model

The prediction model can be retrained with new data:

1. **Prepare Training Data**: Ensure data follows the same feature structure
2. **Feature Engineering**: Update `engineer_features()` function as needed
3. **Model Training**: Use the training scripts in the project
4. **Model Replacement**: Replace the model file in the `models/` directory

### UI Customization

- **Themes**: Modify CSS in templates for provider-specific styling
- **Layout**: Update Bootstrap classes and custom CSS
- **Features**: Add new form fields and data extraction logic

## 🐛 Troubleshooting

### Common Issues

**Application won't start**
- Ensure you're in the `src/` directory when running `python app.py`
- Check that all dependencies are installed
- Verify the ML model file exists in `models/best_model_20250906_043926.pkl`

**Predictions showing default values**
- Restart the application after any changes to `predict.py`
- Ensure the model file is accessible and not corrupted
- Check browser console for JavaScript errors

**Scraping fails**
- Verify the URL is correct and accessible
- Check if the website has changed its structure
- Some sites may require different scraping approaches

**Network access issues**
- Application runs on `0.0.0.0:5000` for network access
- Use `127.0.0.1:5000` or `localhost:5000` for local access only
- Check firewall settings if accessing from other devices

### Performance Tips

- **Large Properties**: Properties with many photos may take longer to load
- **Network Speed**: Scraping speed depends on internet connection
- **Model Loading**: First prediction may be slower due to model loading

## 📈 Future Enhancements

### Planned Features
- **Real-time Market Data**: Integration with property market APIs
- **Comparative Analysis**: Side-by-side property comparisons
- **Price History**: Tracking price changes over time
- **Email Alerts**: Notifications for properties matching criteria
- **Advanced Filters**: Search and filter scraped properties
- **Data Analytics**: Property market trends and insights

### Technical Improvements
- **Database Integration**: Store scraped data persistently
- **Caching System**: Improve performance with Redis/Memcached
- **User Authentication**: Personal property lists and preferences
- **Mobile App**: Native mobile application
- **API Rate Limiting**: Prevent abuse and ensure fair usage

## 📄 License & Credits

This project is developed for educational and research purposes. Please respect website terms of service when scraping and ensure compliance with local regulations.

### Acknowledgments
- **Bootstrap Team** for the responsive framework
- **Font Awesome** for the icon library
- **FastAPI Team** for the excellent web framework
- **scikit-learn** community for the ML tools

---

**Last Updated**: September 6, 2025  
**Version**: 2.0.0  
**Author**: Real Estate Development Team
