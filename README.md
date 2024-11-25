# FundFlow-ML-Backend

This repository contains a Streamlit-based web application designed to perform various tasks including category management, dashboard visualization, Thai text analysis, and a virtual assistant feature.

## Directory Structure

```plaintext
.
├── app
│   ├── data/                                # Data folder
├── models
│   ├── anomaly_detection_0.1.0.h5           # Anomaly detection model
│   ├── anomaly_detection_loader.py          # Anomaly detection model loader
│   ├── word_embedding_0.1.0.bin             # Pre-trained word embedding model
│   ├── word_embedding_loader.py             # Word embedding model loader
├── src
│   ├── api
│   │   ├── api_request_handler.py           # API request handler
│   ├── extraction
│   │   ├── info_extractor.py                # Information extraction utility
│   │   ├── ocr_extractor.py                 # OCR extraction utility
│   │   ├── qr_code_reader.py                # QR code reader utility
│   ├── file
│   │   ├── zip_image_reader.py              # Utility to read images from ZIP files
├── main.py                                  # Main entry point for the project
├── .gitignore                               # Git ignore file
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies

```

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app
   ```
   $ cd app
   ```

   ```
   $ fastapi dev main.py
   ```
