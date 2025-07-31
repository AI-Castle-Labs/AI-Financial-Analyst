from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
from selenium.webdriver.chrome.options import Options
import json
import csv

def setup_driver():
    """Set up and return Chrome WebDriver with options"""
    service = Service()  # ChromeDriver will be auto-detected
    options = Options()
    options.add_argument("--headless")  # Run in headless mode (optional)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def extract_listings(driver, max_listings=20):
    """Extract Airbnb listing information"""
    listings = []
    wait = WebDriverWait(driver, 10)
    
    try:
        # Wait for listings to load - try multiple selectors
        listing_selectors = [
            '[data-testid="card-container"]',
            '[data-testid="listing-card"]',
            '.c4mnd7m',  # Common Airbnb listing container class
            '.g1qv1ctd'  # Another common listing class
        ]
        
        listing_elements = []
        for selector in listing_selectors:
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                listing_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if listing_elements:
                    print(f"Found {len(listing_elements)} listings using selector: {selector}")
                    break
            except TimeoutException:
                continue
        
        if not listing_elements:
            print("No listing containers found with any selector")
            return listings
            
        time.sleep(3)  # Additional wait for dynamic content
        
        for i, listing in enumerate(listing_elements[:max_listings]):
            try:
                listing_data = {}
                
                # Extract title
                try:
                    # Try multiple selectors for title
                    title_element = None
                    title_selectors = [
                        'h1',  # Direct H1 tag within the listing
                        '[data-testid="listing-card-title"]',
                        '.t1jojoys',  # Common Airbnb title class
                        'a[aria-label]'  # Link with aria-label often contains title
                    ]
                    
                    for selector in title_selectors:
                        try:
                            title_element = listing.find_element(By.CSS_SELECTOR, selector)
                            break
                        except NoSuchElementException:
                            continue
                    
                    if title_element:
                        # Get text from aria-label if available, otherwise use text content
                        title = title_element.get_attribute('aria-label') or title_element.text.strip()
                        listing_data['title'] = title
                    else:
                        listing_data['title'] = 'N/A'
                        
                except Exception as e:
                    print(f"Error extracting title: {e}")
                    listing_data['title'] = 'N/A'
                
                # Extract price
                try:
                    price_element = None
                    price_selectors = [
                        '[data-testid="price-and-discounted-price"]',
                        '.pquyp1l',  # Common Airbnb price class
                        '[data-testid="listing-card-subtitle"] span',
                        '.t18uo3af'  # Another price class
                    ]
                    
                    for selector in price_selectors:
                        try:
                            price_element = listing.find_element(By.CSS_SELECTOR, selector)
                            break
                        except NoSuchElementException:
                            continue
                    
                    if price_element:
                        listing_data['price'] = price_element.text.strip()
                    else:
                        listing_data['price'] = 'N/A'
                        
                except Exception as e:
                    print(f"Error extracting price: {e}")
                    listing_data['price'] = 'N/A'
                
                # Extract rating
                try:
                    rating_element = listing.find_element(By.CSS_SELECTOR, '[data-testid="listing-card-subtitle"]')
                    listing_data['rating'] = rating_element.text.strip()
                except NoSuchElementException:
                    listing_data['rating'] = 'N/A'
                
                # Extract image URL
                try:
                    image_element = listing.find_element(By.CSS_SELECTOR, 'img')
                    listing_data['image_url'] = image_element.get_attribute('src')
                except NoSuchElementException:
                    listing_data['image_url'] = 'N/A'
                
                # Extract link
                try:
                    link_element = listing.find_element(By.CSS_SELECTOR, 'a')
                    listing_data['link'] = link_element.get_attribute('href')
                except NoSuchElementException:
                    listing_data['link'] = 'N/A'
                
                if listing_data['title'] != 'N/A':  # Only add if we got at least a title
                    listings.append(listing_data)
                    print(f"Extracted listing {i+1}: {listing_data['title']}")
                
            except Exception as e:
                print(f"Error extracting listing {i+1}: {str(e)}")
                continue
                
    except TimeoutException:
        print("Timeout waiting for listings to load")
    except Exception as e:
        print(f"Error finding listings: {str(e)}")
    
    return listings

def save_to_csv(listings, filename='airbnb_listings.csv'):
    """Save listings to CSV file"""
    if not listings:
        print("No listings to save")
        return
    
    fieldnames = ['title', 'price', 'rating', 'image_url', 'link']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(listings)
    
    print(f"Saved {len(listings)} listings to {filename}")

def save_to_json(listings, filename='airbnb_listings.json'):
    """Save listings to JSON file"""
    if not listings:
        print("No listings to save")
        return
    
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(listings, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(listings)} listings to {filename}")

def main():
    """Main function to run the travel agent"""
    driver = None
    
    try:
        print("Starting Airbnb Travel Agent...")
        driver = setup_driver()
        
        print("Navigating to Airbnb...")
        driver.get("https://www.airbnb.com/s/Istanbul--Turkey/homes")
        
        print("Extracting listings...")
        listings = extract_listings(driver, max_listings=20)
        
        if listings:
            print(f"\nFound {len(listings)} listings:")
            for i, listing in enumerate(listings, 1):
                print(f"\n{i}. {listing['title']}")
                print(f"   Price: {listing['price']}")
                print(f"   Rating: {listing['rating']}")
            
            # Save data
            save_to_csv(listings)
            save_to_json(listings)
        else:
            print("No listings found")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        if driver:
            print("Closing browser...")
            driver.quit()

if __name__ == "__main__":
    main()
