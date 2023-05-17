# tcp-server

A simple tcp server framework inspired by flask.

## Usage

1. Prepare a json to claim data format.
    ```python
    config_file = "demo_config.json"
    ```
2. Code your own business functions and then register them with decorators:
    ```python
    app = TCPApp(config=config_file)
    @app.function(id=1)
    def test(*, x):
        # Do something
        return {"y": x + 1000}

    ```
    you can also registere error handlers
    ```python
    @app.error_handler(exc=TypeError, code=1)
    def handle(e):
        # Do something
        return 16, "type error"
    ```
3. Start your server:
    ```python
    app.run()
    ```

4. Communicate the server with cilent in another process:

```python
    from tcp_server.server import TestClient

    host = "localhost"
    config_file = "demo_config.json"

    client = TestClient(host=host, config=config_file)
    test_request = {
        "x": 2,
        "request_no": 1,
        "instruction_id": 1,
    }
    client.test(test_request)
```
