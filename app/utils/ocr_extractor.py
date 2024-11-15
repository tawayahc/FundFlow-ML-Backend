import easyocr
import numpy as np
from .qr_code_reader import QRCodeReader

class OCRExtractor:
    def __init__(self, languages=['en', 'th'], gpu=True):
        """
        Initializes the OCRExtractor with specified languages and GPU settings.

        :param languages: List of languages for OCR.
        :param gpu: Boolean flag to use GPU.
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text_from_images(self, images, categories):
        """
        Extract text from a list of images.

        :param images: List of PIL Image objects.
        :return: List of extracted text.
        """
        extracted_text = []
        qr_reader = QRCodeReader()

        for img in images:
            decoded_text = qr_reader.read_qr_code(img)
            if (decoded_text is not None) and (decoded_text not in categories):
                img_np = np.array(img)
                result = self.reader.readtext(img_np, detail=0)
                data = {
                    "decoded_text": decoded_text,
                    "ocr_text": result
                }
                extracted_text.append(data)
            else:
                continue

        return extracted_text
