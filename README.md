# FundFlow-ML-Backend

This project is a robust backend system built using FastAPI that provides API for data extraction, processing, and machine learning model integration. It is designed to handle various functionalities such as Optical Character Recognition (OCR), QR code reading, information extraction, and machine learning model predictions.

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
3. Access the API documentation Visit http://127.0.0.1:8000/docs for the Swagger UI

## API Endpoint

### `/get_transaction` [POST]
This endpoint processes an uploaded ZIP file containing images, detect only payment slip image to extracts relevant information using OCR and QR code reading, find relevant category by calculating cosine similarity implement with word embedding model, and returns structured transaction data.

**Request Parameters**
- `file`: Uploaded ZIP file containing images
- `header`: Dictionary containing the bearer token for authentication

***Example of Parameter header***
```json
{
    "Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6IiIsInVzZXJfaWQiOjEsImV4cCI6MTczMjYwOTkwOX0.kdedIkkE8HFGH_XRIeO_tsis-D_JzZVGdJiYOPZgll4"
}
```

**Response**
- A dictionary containing the extracted transaction information for each image

***Example of Response***
```json
[
   {
       "metadata": "0041000600000101030040220014158193429APM020415102TH91042BBB",
       "bank": "ธนาคารกสิกรไทย",
       "type": "expense",
       "amount": 22500.0,
       "category_id": 1,
       "date": "2024-06-06",
       "time": "19:34:00.000000",
       "memo": "ค่าเทอม"
   }
]
```



