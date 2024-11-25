import re
from datetime import datetime

class InfoExtractor:
    def __init__(self):
        self.patterns = {
            'bank': r"(kbank)",
            "amount": r"amount:*\s*([\d,ol]+\.?[\d,ol]*)\s*baht",
            "fee": r"(fee:?|baht)\s*([\d,ol]+\.?[\d,ol]*)\s*baht",
            "date": r"(\d{1,2}(\s[\w\|\$]{3})?\s\d{2,4})",
            "time": r"(\d{1,2}:\d{2}\s[ap]\.?[m\[]\.?|\d{1,2}:\d{2})",
            "memo": r"memo:\s*(.*)"
        }

    def clean_text(self, text):
        """
        Clean the input text by normalizing whitespace and removing unwanted characters.

        :param text: The raw input text to be cleaned.
        :return: A cleaned version of the input text with normalized spaces and removed special characters.
        """
        cleaned_text = re.sub(r'\s+', ' ', text.strip())   # Normalize multiple spaces to a single space.
        cleaned_text = cleaned_text.replace('\u200b', '')  # Remove zero-width space characters.
        return cleaned_text
    
    def correct_amount_fee(self, value):
        """
        Correct common OCR errors in numeric values, such as misinterpreted characters ('o' as '0' and 'l' as '1').
        Removes commas from the values and converts them to a float.

        :param value: The string value to correct (e.g., extracted amount or fee).
        :return: The corrected value as a float, or 0.0 if the value is None or invalid.
        """
        if value is None:
            return 0.0
        
        if isinstance(value, float):
            return value

        # Replace common OCR errors
        corrected_value = value.replace('o', '0')
        corrected_value = corrected_value.replace('l', '1')
        corrected_value = corrected_value.replace(',', '')

        # Attempt to convert the corrected string to a float
        try:
            corrected_value = float(corrected_value)
        except ValueError:
            corrected_value = None
        return corrected_value
    
    def correct_bank(self, bank):
        if bank is None:
            return None

        match bank:
            case "kbank":
                return "ธนาคารกสิกรไทย"
            case "scb":
                return "ธนาคารไทยพาณิชย์"
            case _:
                return None
            
    def parse_date_time(self, date_str, time_str):
        """
        Parse the extracted date and time strings into the desired formats separately.

        :param date_str: The date string (e.g., '4 mar 24').
        :param time_str: The time string (e.g., '6:23 pm').
        :return: A tuple containing formatted date and time strings or None for each if parsing fails.
        """
        def parse_date(date_str):
            try:
                date_obj = datetime.strptime(date_str, "%d %b %y").date()
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                return None

        def parse_time(time_str):
            try:
                time_obj = datetime.strptime(time_str, "%I:%M %p").time()
                return time_obj.strftime("%H:%M:%S.%f")
            except ValueError:
                return None

        formatted_date = parse_date(date_str) if date_str else None
        formatted_time = parse_time(time_str) if time_str else None

        return formatted_date, formatted_time


    def extract_info(self, text):
        """
        Extract payment-related information from a given text, including date, amount, fee, and memo,
        using pre-defined regular expressions.

        :param text: The raw input text containing payment details.
        :return: A dictionary with keys 'Date', 'Amount', 'Fee', and 'Memo', containing the extracted values
                 or None if a particular field is not found.
        """
        cleaned_text = self.clean_text(text)
        extracted_info = {}

        for key, pattern in self.patterns.items():
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                if key == 'fee':
                    extracted_value = match.group(2).strip()
                else:
                    extracted_value = match.group(1).strip()

                if key in ['amount', 'fee']:
                    extracted_value = self.correct_amount_fee(extracted_value)

                if key == 'bank':
                    extracted_value = self.correct_bank(extracted_value)

                extracted_info[key] = extracted_value
            else:
                extracted_info[key] = None

        extracted_date = extracted_info.get('date')
        extracted_time = extracted_info.get('time')
        formatted_date, formatted_time = self.parse_date_time(extracted_date, extracted_time)
        extracted_info['date'] = formatted_date
        extracted_info['time'] = formatted_time

        return extracted_info
