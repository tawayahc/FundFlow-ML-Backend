import requests
from typing import Dict, Any, Optional

class APIRequestHandler:
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 10):
        """
        Initialize the API request handler.
        
        :param base_url: The base URL of the API.
        :param headers: Default headers to include in every request (optional).
        :param timeout: Default timeout for requests (in seconds).
        """
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Perform a GET request to the API.

        :param endpoint: The API endpoint (relative to the base URL).
        :param params: Query parameters for the GET request.
        :return: The parsed JSON response or an error dictionary.
        """
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
