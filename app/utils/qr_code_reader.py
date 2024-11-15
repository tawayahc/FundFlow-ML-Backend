import cv2
import numpy as np

class QRCodeReader:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def read_qr_code(self, image):
        """
        Read QR code from a list of images.

        :param images: A PIL Image objects.
        :return: QR code data.
        """
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        decoded_object, _, _ = self.detector.detectAndDecode(img_np)

        if decoded_object:
            return decoded_object
        else:
            return None
