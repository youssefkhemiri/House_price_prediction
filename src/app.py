from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
from datetime import datetime
from scrapers.menzili_scraper import scrape_menzili
from scrapers.any_website_scraper import scrape_any_website
from scrapers.mubawab_scraper import scrape_mubawab
import traceback
import asyncio
from typing import Optional

# Import predict function (placeholder for now)
try:
    from predict import predict_price
except ImportError:
    def predict_price(property_data: dict) -> dict:
        """Placeholder predict function - returns mock prediction"""
        return {
            "predicted_price": 450000,
            "confidence": 0.85,
            "price_range": {"min": 400000, "max": 500000},
            "factors": ["location", "size", "features"]
        }

app = FastAPI(
    title="Real Estate Scraper",
    description="Extract property data from real estate websites",
    version="1.0.0"
)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Mount static files (create directory if it doesn't exist)
static_dir = os.path.join(current_dir, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Custom template filters
def format_number(value):
    """Format numbers with commas"""
    try:
        return "{:,}".format(int(value))
    except (ValueError, TypeError):
        return value

def format_date(timestamp_ms):
    """Convert millisecond timestamp to formatted date"""
    try:
        if timestamp_ms:
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            return dt.strftime('%B %d, %Y')
        return 'Date not available'
    except (ValueError, TypeError):
        return 'Date not available'

# Add custom filters to Jinja2 environment
templates.env.filters['format_number'] = format_number
templates.env.filters['format_date'] = format_date

# Supported website providers
SUPPORTED_PROVIDERS = {
    'menzili': {
        'name': 'Menzili',
        'domain': 'menzili.tn',
        'description': 'Tunisian real estate platform',
        'icon': 'building'
    },
    'mubawab': {
        'name': 'Mubawab',
        'domain': 'mubawab.tn',
        'description': 'Tunisian real estate platform',
        'icon': 'home'
    },
    'any_website': {
        'name': 'Any Website',
        'domain': 'any',
        'description': 'Universal property scraper',
        'icon': 'globe'
    }
}

# Pydantic models
class ScrapeRequest(BaseModel):
    provider: str
    url: str

class ScrapeResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    prediction: Optional[dict] = None
    error: Optional[str] = None

class PredictionRequest(BaseModel):
    property_data: dict

class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[dict] = None
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with the scraper form"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "providers": SUPPORTED_PROVIDERS
    })

@app.get("/manual", response_class=HTMLResponse)
async def manual_entry(request: Request):
    """Manual property entry form"""
    return templates.TemplateResponse("manual_entry.html", {"request": request})

@app.post("/scrape", response_class=HTMLResponse)
async def scrape_form(
    request: Request,
    provider: str = Form(...),
    url: str = Form(...)
):
    """Handle the scraping request from form"""
    try:
        # Validate provider
        if provider not in SUPPORTED_PROVIDERS:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "providers": SUPPORTED_PROVIDERS,
                "error": "Please select a valid provider"
            })
        
        # Validate URL
        url = url.strip()
        if not url:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "providers": SUPPORTED_PROVIDERS,
                "error": "Please provide a URL to scrape"
            })
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Check if URL matches the selected provider (skip for any_website)
        if provider != 'any_website':
            provider_domain = SUPPORTED_PROVIDERS[provider]['domain']
            if provider_domain not in url:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "providers": SUPPORTED_PROVIDERS,
                    "error": f"URL does not match the selected provider ({provider_domain})"
                })
        
        # Scrape based on provider
        if provider == 'menzili':
            data = scrape_menzili(url)
        elif provider == 'mubawab':
            data = scrape_mubawab(url)
        elif provider == 'any_website':
            data = scrape_any_website(url)
        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "providers": SUPPORTED_PROVIDERS,
                "error": "Scraper for this provider is not implemented yet"
            })
        
        # Get price prediction
        try:
            prediction = predict_price(data)
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = None
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "data": data,
            "provider": SUPPORTED_PROVIDERS[provider],
            "prediction": prediction
        })
        
    except Exception as e:
        error_msg = f"Error scraping the URL: {str(e)}"
        print(f"Scraping error: {traceback.format_exc()}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "providers": SUPPORTED_PROVIDERS,
            "error": error_msg
        })

@app.post("/manual-entry", response_class=HTMLResponse)
async def manual_form_submit(
    request: Request,
    property_type: str = Form(...),
    price: Optional[str] = Form(None),
    currency: str = Form(default="DT"),
    transaction_type: str = Form(...),
    address: Optional[str] = Form(None),
    governorate: Optional[str] = Form(None),
    delegation: Optional[str] = Form(None),
    locality: Optional[str] = Form(None),
    living_area: Optional[str] = Form(None),
    land_area: Optional[str] = Form(None),
    room_count: Optional[str] = Form(None),
    bathroom_count: Optional[str] = Form(None),
    construction_year: Optional[str] = Form(None),
    floor: Optional[str] = Form(None),
    has_garage: bool = Form(default=False),
    has_garden: bool = Form(default=False),
    has_pool: bool = Form(default=False),
    has_balcony: bool = Form(default=False),
    has_terrace: bool = Form(default=False),
    heating: bool = Form(default=False),
    air_conditioning: bool = Form(default=False),
    furnished: bool = Form(default=False),
    condition: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Handle manual property entry form submission"""
    try:
        # Convert form data to property data structure
        data = {
            "url": "manual_entry",
            "id": f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "source": "manual",
            "property_type": property_type,
            "price": price if price and price.strip() else "nan",
            "currency": currency,
            "description": [description] if description and description.strip() else [],
            "address": address if address and address.strip() else None,
            "governorate": governorate if governorate and governorate.strip() else "nan",
            "delegation": delegation if delegation and delegation.strip() else None,
            "locality": locality if locality and locality.strip() else None,
            "postal_code": None,
            "living_area": living_area if living_area and living_area.strip() else "nan",
            "land_area": int(land_area) if land_area and land_area.strip() and land_area.isdigit() else None,
            "room_count": int(room_count) if room_count and room_count.strip() and room_count.isdigit() else None,
            "bathroom_count": int(bathroom_count) if bathroom_count and bathroom_count.strip() and bathroom_count.isdigit() else None,
            "construction_year": int(construction_year) if construction_year and construction_year.strip() and construction_year.isdigit() else None,
            "floor": int(floor) if floor and floor.strip() and floor.isdigit() else None,
            "has_garage": has_garage,
            "has_garden": has_garden,
            "has_pool": has_pool,
            "has_balcony": has_balcony,
            "has_terrace": has_terrace,
            "heating": heating,
            "air_conditioning": air_conditioning,
            "furnished": furnished,
            "phone": [],
            "agency": None,
            "contact_name": None,
            "listing_date": int(datetime.now().timestamp() * 1000),
            "last_updated": None,
            "photos": [],
            "features": [],
            "condition": condition if condition and condition.strip() else None,
            "transaction_type": transaction_type,
        }
        
        # Get price prediction
        try:
            prediction = predict_price(data)
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = None
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "data": data,
            "provider": {"name": "Manual Entry", "icon": "fas fa-edit"},
            "prediction": prediction
        })
        
    except Exception as e:
        error_msg = f"Error processing manual entry: {str(e)}"
        print(f"Manual entry error: {traceback.format_exc()}")
        return templates.TemplateResponse("manual_entry.html", {
            "request": request,
            "error": error_msg
        })

@app.post("/api/scrape", response_model=ScrapeResponse)
async def api_scrape(scrape_request: ScrapeRequest):
    """API endpoint for scraping (returns JSON)"""
    try:
        provider = scrape_request.provider
        url = scrape_request.url.strip()
        
        if provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(status_code=400, detail="Invalid provider")
        
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Scrape based on provider
        if provider == 'menzili':
            data = scrape_menzili(url)
        elif provider == 'mubawab':
            data = scrape_mubawab(url)
        elif provider == 'any_website':
            data = scrape_any_website(url)
        else:
            raise HTTPException(status_code=400, detail="Scraper not implemented for this provider")
        
        # Get price prediction
        try:
            prediction = predict_price(data)
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = None
        
        return ScrapeResponse(success=True, data=data, prediction=prediction)
        
    except HTTPException:
        raise
    except Exception as e:
        return ScrapeResponse(success=False, error=str(e))

@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(prediction_request: PredictionRequest):
    """API endpoint for price prediction only"""
    try:
        prediction = predict_price(prediction_request.property_data)
        return PredictionResponse(success=True, prediction=prediction)
    except Exception as e:
        return PredictionResponse(success=False, error=str(e))

@app.get("/docs")
async def get_docs():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
