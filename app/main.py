from fastapi import FastAPI, UploadFile, File, HTTPException
from utils.zip_image_reader import ZipImageReader
from utils.ocr_extractor import OCRExtractor
import os

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/extract_text")
async def extract_text_from_zip(file: UploadFile = File(...)):
    """
    Extract text from images inside a ZIP file.

    :param file: Uploaded ZIP file containing images.
    :return: Dictionary containing text extracted from each image.
    """
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a ZIP archive.")
    
    # Save the uploaded ZIP file temporarily
    zip_filepath = f"data/{file.filename}"
    with open(zip_filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Initialising the ZipImageReader and OCRExtractor
    zip_reader = ZipImageReader(zip_filepath)
    ocr_extractor = OCRExtractor(languages=['en', 'th'], gpu=True)
    
    # Extract images from the ZIP file
    images = zip_reader.extract_images()

    if not images:
        raise HTTPException(status_code=400, detail="No images found in the ZIP file.")
    
    # Extract text from the images
    extracted_text = ocr_extractor.extract_text_from_images(images)

    # Clean up the temporary ZIP file
    os.remove(zip_filepath)

    result = {f"image_{i + 1}": text for i, text in enumerate(extracted_text)}

    return {"extracted_text": result}
