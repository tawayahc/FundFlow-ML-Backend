import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.api.api_request_handler import APIRequestHandler
from src.file.zip_image_reader import ZipImageReader
from src.extraction.ocr_extractor import OCRExtractor
from src.extraction.qr_code_reader import QRCodeReader
from src.extraction.info_extractor import InfoExtractor
from models.anomaly_detection_loader import AnomalyDetectionModelLoader
from models.word_embedding_loader import FastTextSimilarity

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/get_transaction")
async def get_transaction(file: UploadFile = File(...), bearer_token: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6IiIsInVzZXJfaWQiOjEsImV4cCI6MTczMjYwOTkwOX0.kdedIkkE8HFGH_XRIeO_tsis-D_JzZVGdJiYOPZgll4'):
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a ZIP archive.")
    
    # Save the uploaded ZIP file temporarily
    zip_filepath = f"data/{file.filename}"
    with open(zip_filepath, "wb") as buffer:
        buffer.write(await file.read())

    zip_reader = ZipImageReader(zip_filepath)
    
    # Extract images from the ZIP file
    images = zip_reader.extract_images()
    if not images:
        raise HTTPException(status_code=400, detail="No images found in the ZIP file.")

    # Detect anomalies in the images
    anomaly_detection_model = AnomalyDetectionModelLoader()
    anomaly_detection_model.load_model()
    slip_images = anomaly_detection_model.predict_anomaly(images)

    # Decode QR codes from the images
    qr_reader = QRCodeReader()
    decoded_texts = [qr_reader.read_qr_code(img) for img in slip_images]

    # Get the transaction history from the API
    api_handler = APIRequestHandler(base_url="http://localhost:8080", headers={"Authorization": f"Bearer {bearer_token}"})
    transactions_history = api_handler.get("transactions/all")
    metadata_history = [th['meta_data'] for th in transactions_history]

    # Get the index of unique metadata
    unique_index = [index for index, metadata in enumerate(decoded_texts) if metadata not in metadata_history]

    # Get the unique metadata and slip images
    metadata_list = [decoded_texts[index] for index in unique_index]
    image_list = [slip_images[index] for index in unique_index]

    # Extract text from the images
    ocr_extractor = OCRExtractor(languages=['en', 'th'], gpu=True)
    extracted_texts = ocr_extractor.extract_text_from_images(image_list)

    # Get the categories from the API
    categories = api_handler.get("categories/all")
    categories_name = [category['name'] for category in categories[1:]]

    get_info = InfoExtractor()
    word_embedding_model = FastTextSimilarity()
    word_embedding_model.load_model()
    response = []
    for text, metadata in zip(extracted_texts, metadata_list):
        text = " ".join(text)
        info = get_info.extract_info(text)
        category, _ = word_embedding_model.find_most_similar(info['memo'], categories_name)

        transaction = {
            "metadata": metadata,
            "bank": info['bank'],
            "type": "expense",
            "amount": info['amount'] + info['fee'],
            "category_id": categories_name.index(category) if category in categories_name else -1,
            "date": info['date'],
            "time": info['time'],
            "memo": info['memo']
        }
        response.append(transaction)

    # Clean up the temporary ZIP file
    os.remove(zip_filepath)

    return response
