import requests
from src.STATIC import TARGET_IP, PORT, API_KEY

def test():
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/test_connection",
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    print(response.json())

    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/test_data_streaming",
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    print(response.json())


if __name__ == '__main__':
    test()