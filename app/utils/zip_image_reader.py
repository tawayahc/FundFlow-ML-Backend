import zipfile
from PIL import Image
from io import BytesIO

class ZipImageReader:
    def __init__(self, zip_filepath):
        """
        Initialize the ZipImageReader with the path to a ZIP file.

        :param zip_filepath: Path to the ZIP file.
        """
        self.zip_filepath = zip_filepath

    def extract_images(self):
        """
        Extract images from the ZIP file and return them as a list.

        :return: List of PIL Image objects.
        """
        images = []
        try:
            with zipfile.ZipFile(self.zip_filepath, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Skip directories, macOS metadata files, and non-image files
                    if file_info.is_dir() or file_info.filename.startswith('__MACOSX/') or not file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        continue

                    with zip_ref.open(file_info) as file:
                        try:
                            img = Image.open(BytesIO(file.read()))
                            img.load()  # Ensure the image is fully loaded
                            images.append(img)
                        except (IOError, OSError):
                            print(f"Error opening image: {file_info.filename}")
            
            print(f"Successfully extracted and read {len(images)} images.")

        except zipfile.BadZipFile:
            print(f"Error: Invalid zip file: {self.zip_filepath}")
        except FileNotFoundError:
            print(f"Error: Zip file not found: {self.zip_filepath}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return images
