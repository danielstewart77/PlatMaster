"""
Test script for PlatMaster FastAPI server.
Copyright (C) 2025 Daniel Stewart

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import requests
import os

def test_api():
    """Test the PlatMaster API with a sample PDF file."""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code} - {response.json()}")
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    response = requests.get(base_url)
    print(f"Root endpoint: {response.status_code} - {response.json()}")
    
    # Test extraction endpoint with a PDF file
    plats_dir = "plats"
    if os.path.exists(plats_dir):
        pdf_files = [f for f in os.listdir(plats_dir) if f.lower().endswith('.pdf')]
        if pdf_files:
            test_pdf = os.path.join(plats_dir, pdf_files[0])
            print(f"\nTesting extraction with {test_pdf}...")
            
            with open(test_pdf, 'rb') as f:
                files = {'file': (pdf_files[0], f, 'application/pdf')}
                response = requests.post(f"{base_url}/extract", files=files)
                
            print(f"Extraction result: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("Extraction successful!")
                print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            else:
                print(f"Error: {response.text}")
        else:
            print("No PDF files found in plats directory for testing")
    else:
        print("Plats directory not found - cannot test extraction endpoint")

if __name__ == "__main__":
    test_api()
