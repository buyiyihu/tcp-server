"""This is the demo client for the TCP service framework.

Author: buyiyihu
"""
from tcp_server.server import TestClient

host = "localhost"
config_file = "demo_config.json"

client = TestClient(host=host, config=config_file)

if __name__ == "__main__":
    test_request = {
        "x": 2,
        "request_no": 1,
        "instruction_id": 1,
    }
    client.test(test_request)
