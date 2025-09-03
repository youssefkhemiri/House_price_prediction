#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery application for running house price prediction crawlers on a schedule.

This script sets up Celery tasks to run the Menzili and Mubawab crawlers daily,
collecting real estate listings automatically and saving them to JSON files.

Requirements:
- pip install celery redis
- Redis server running (or another message broker)

Usage:
1. Start Redis server
2. Start Celery worker: celery -A celery_timer worker --loglevel=info
3. Start Celery beat scheduler: celery -A celery_timer beat --loglevel=info
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger

# Add the project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_task_logger(__name__)

# Celery configuration
app = Celery('house_price_crawlers')

# Configure Celery
app.conf.update(
    broker_url='redis://localhost:6379/0',  # Redis broker
    result_backend='redis://localhost:6379/0',  # Redis result backend
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Africa/Tunis',  # Tunisia timezone
    enable_utc=True,
    
    # Task routing
    task_routes={
        'celery_timer.crawl_menzili_task': {'queue': 'crawlers'},
        'celery_timer.crawl_mubawab_task': {'queue': 'crawlers'},
        'celery_timer.daily_crawler_job': {'queue': 'scheduler'},
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# Schedule configuration - runs daily at 2:00 AM Tunisia time
app.conf.beat_schedule = {
    'daily-crawl-all-sites': {
        'task': 'celery_timer.daily_crawler_job',
        'schedule': crontab(hour=2, minute=0),  # 2:00 AM daily
        'options': {'queue': 'scheduler'}
    },
    'weekly-maintenance': {
        'task': 'celery_timer.cleanup_old_data',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # 3:00 AM every Sunday
        'options': {'queue': 'scheduler'}
    },
}

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
MENZILI_JSON_PATH = os.path.join(DATA_DIR, 'menzili_listings.json')
MUBAWAB_JSON_PATH = os.path.join(DATA_DIR, 'mubawab_listings.json')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


@app.task(bind=True, max_retries=3, default_retry_delay=300)
def crawl_menzili_task(self) -> Dict[str, Any]:
    """
    Celery task to run the Menzili crawler.
    
    Returns:
        Dict containing crawl results and statistics
    """
    try:
        logger.info("Starting Menzili crawler task")
        
        # Import the crawler function
        try:
            from scripts.crawlers.menzili_crawler import crawl_menzili_latest
        except ImportError as e:
            logger.error(f"Failed to import Menzili crawler: {e}")
            raise
        
        # Run the crawler
        start_time = datetime.now()
        
        result = crawl_menzili_latest(
            pages_to_scan=5,  # Scan more pages for daily run
            out_json_path=MENZILI_JSON_PATH,
            polite_sleep_range=(1.5, 3.0)  # Be more polite in automated runs
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Add metadata to result
        result.update({
            'task_name': 'menzili_crawler',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'success': True
        })
        
        logger.info(f"Menzili crawler completed successfully: {result['scraped_new']} new listings")
        
        # Save task log
        save_task_log('menzili', result)
        
        return result
        
    except Exception as exc:
        logger.error(f"Menzili crawler task failed: {exc}")
        
        # Log the failure
        error_result = {
            'task_name': 'menzili_crawler',
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': str(exc),
            'retry_count': self.request.retries
        }
        save_task_log('menzili', error_result)
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying Menzili crawler task in {self.default_retry_delay} seconds")
            raise self.retry(countdown=self.default_retry_delay, exc=exc)
        
        raise exc


@app.task(bind=True, max_retries=3, default_retry_delay=300)
def crawl_mubawab_task(self) -> Dict[str, Any]:
    """
    Celery task to run the Mubawab crawler.
    
    Returns:
        Dict containing crawl results and statistics
    """
    try:
        logger.info("Starting Mubawab crawler task")
        
        # Import the crawler
        try:
            from scripts.crawlers.mubaweb_crawler import main as run_mubawab_crawler
            # We need to adapt the mubawab crawler to return results like menzili
            # For now, we'll run it and parse the output
        except ImportError as e:
            logger.error(f"Failed to import Mubawab crawler: {e}")
            raise
        
        start_time = datetime.now()
        
        # Count existing items before crawling
        existing_count = 0
        if os.path.exists(MUBAWAB_JSON_PATH):
            try:
                with open(MUBAWAB_JSON_PATH, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_count = len(existing_data) if isinstance(existing_data, list) else 0
            except Exception:
                existing_count = 0
        
        # Run the crawler - we need to modify the main function to return results
        # For now, let's create a custom runner
        result = run_mubawab_crawler_with_results()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Add metadata to result
        result.update({
            'task_name': 'mubawab_crawler',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'success': True
        })
        
        logger.info(f"Mubawab crawler completed successfully: {result.get('scraped_new', 0)} new listings")
        
        # Save task log
        save_task_log('mubawab', result)
        
        return result
        
    except Exception as exc:
        logger.error(f"Mubawab crawler task failed: {exc}")
        
        # Log the failure
        error_result = {
            'task_name': 'mubawab_crawler',
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': str(exc),
            'retry_count': self.request.retries
        }
        save_task_log('mubawab', error_result)
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying Mubawab crawler task in {self.default_retry_delay} seconds")
            raise self.retry(countdown=self.default_retry_delay, exc=exc)
        
        raise exc


def run_mubawab_crawler_with_results() -> Dict[str, Any]:
    """
    Run the Mubawab crawler and return results in a consistent format.
    
    This function adapts the existing mubawab_crawler.main() to return
    structured results similar to the Menzili crawler.
    """
    import subprocess
    import json
    from pathlib import Path
    
    # Count existing items before
    existing_count = 0
    if os.path.exists(MUBAWAB_JSON_PATH):
        try:
            with open(MUBAWAB_JSON_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_count = len(existing_data) if isinstance(existing_data, list) else 0
        except Exception:
            existing_count = 0
    
    # Run the crawler script
    crawler_script = os.path.join(PROJECT_ROOT, 'scripts', 'crawlers', 'mubaweb_crawler.py')
    
    try:
        # Run the crawler as a subprocess to capture any output
        result = subprocess.run(
            [sys.executable, crawler_script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"Mubawab crawler script failed: {result.stderr}")
        
        # Count items after
        final_count = 0
        if os.path.exists(MUBAWAB_JSON_PATH):
            try:
                with open(MUBAWAB_JSON_PATH, 'r', encoding='utf-8') as f:
                    final_data = json.load(f)
                    final_count = len(final_data) if isinstance(final_data, list) else 0
            except Exception:
                final_count = existing_count
        
        scraped_new = max(0, final_count - existing_count)
        
        return {
            'pages_scanned': 3,  # Default from the crawler
            'found_urls': scraped_new,  # Approximation
            'scraped_new': scraped_new,
            'total_saved': final_count,
            'output_file': MUBAWAB_JSON_PATH,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        logger.error(f"Failed to run Mubawab crawler: {e}")
        raise


@app.task(bind=True)
def daily_crawler_job(self):
    """
    Main daily task that orchestrates both crawlers.
    
    This task runs both Menzili and Mubawab crawlers and provides
    a summary of the overall daily crawling results.
    """
    try:
        logger.info("Starting daily crawler job")
        
        start_time = datetime.now()
        results = {}
        
        # Run Menzili crawler
        try:
            logger.info("Executing Menzili crawler task")
            menzili_result = crawl_menzili_task.apply_async()
            results['menzili'] = menzili_result.get(timeout=1800)  # 30 minutes timeout
        except Exception as e:
            logger.error(f"Menzili crawler failed: {e}")
            results['menzili'] = {'success': False, 'error': str(e)}
        
        # Run Mubawab crawler
        try:
            logger.info("Executing Mubawab crawler task")
            mubawab_result = crawl_mubawab_task.apply_async()
            results['mubawab'] = mubawab_result.get(timeout=1800)  # 30 minutes timeout
        except Exception as e:
            logger.error(f"Mubawab crawler failed: {e}")
            results['mubawab'] = {'success': False, 'error': str(e)}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create summary
        summary = {
            'job_name': 'daily_crawler_job',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'results': results,
            'total_new_listings': (
                results.get('menzili', {}).get('scraped_new', 0) +
                results.get('mubawab', {}).get('scraped_new', 0)
            ),
            'success': all(r.get('success', False) for r in results.values())
        }
        
        logger.info(f"Daily crawler job completed: {summary['total_new_listings']} total new listings")
        
        # Save daily summary
        save_daily_summary(summary)
        
        return summary
        
    except Exception as exc:
        logger.error(f"Daily crawler job failed: {exc}")
        raise


@app.task
def cleanup_old_data():
    """
    Weekly maintenance task to clean up old log files and optimize data files.
    """
    try:
        logger.info("Starting weekly maintenance task")
        
        # Clean up old log files (older than 30 days)
        log_files_cleaned = 0
        if os.path.exists(LOG_DIR):
            for filename in os.listdir(LOG_DIR):
                file_path = os.path.join(LOG_DIR, filename)
                if os.path.isfile(file_path):
                    # Check file age
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_age > timedelta(days=30):
                        os.remove(file_path)
                        log_files_cleaned += 1
        
        # Backup data files
        backup_data_files()
        
        result = {
            'task_name': 'cleanup_old_data',
            'timestamp': datetime.now().isoformat(),
            'log_files_cleaned': log_files_cleaned,
            'success': True
        }
        
        logger.info(f"Weekly maintenance completed: {log_files_cleaned} old log files cleaned")
        save_task_log('maintenance', result)
        
        return result
        
    except Exception as exc:
        logger.error(f"Weekly maintenance failed: {exc}")
        raise


def save_task_log(task_type: str, result: Dict[str, Any]) -> None:
    """Save task execution log to file."""
    try:
        log_file = os.path.join(LOG_DIR, f"{task_type}_tasks.jsonl")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to save task log: {e}")


def save_daily_summary(summary: Dict[str, Any]) -> None:
    """Save daily summary to file."""
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        summary_file = os.path.join(LOG_DIR, f"daily_summary_{date_str}.json")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save daily summary: {e}")


def backup_data_files() -> None:
    """Create backup copies of the main data files."""
    try:
        from shutil import copy2
        
        backup_dir = os.path.join(DATA_DIR, 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Backup Menzili data
        if os.path.exists(MENZILI_JSON_PATH):
            backup_path = os.path.join(backup_dir, f"menzili_listings_{timestamp}.json")
            copy2(MENZILI_JSON_PATH, backup_path)
        
        # Backup Mubawab data
        if os.path.exists(MUBAWAB_JSON_PATH):
            backup_path = os.path.join(backup_dir, f"mubawab_listings_{timestamp}.json")
            copy2(MUBAWAB_JSON_PATH, backup_path)
            
        logger.info(f"Data files backed up with timestamp {timestamp}")
        
    except Exception as e:
        logger.error(f"Failed to backup data files: {e}")


# CLI commands for manual task execution
@app.task
def run_menzili_now():
    """Manual task to run Menzili crawler immediately."""
    return crawl_menzili_task.delay()


@app.task
def run_mubawab_now():
    """Manual task to run Mubawab crawler immediately."""
    return crawl_mubawab_task.delay()


@app.task
def run_both_crawlers_now():
    """Manual task to run both crawlers immediately."""
    return daily_crawler_job.delay()


if __name__ == '__main__':
    # This allows running the script directly for testing
    print("House Price Prediction Crawlers - Celery Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Log Directory: {LOG_DIR}")
    print()
    print("To start the system:")
    print("1. Start Redis server")
    print("2. Start Celery worker: celery -A celery_timer worker --loglevel=info")
    print("3. Start Celery beat: celery -A celery_timer beat --loglevel=info")
    print()
    print("Manual task execution:")
    print("- celery -A celery_timer call celery_timer.run_menzili_now")
    print("- celery -A celery_timer call celery_timer.run_mubawab_now")
    print("- celery -A celery_timer call celery_timer.run_both_crawlers_now")
