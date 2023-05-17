"""A test client for TCP server .

Author: buyiyihu

Usage:
  python3 client.py <hexified bytes>

  if <hexified bytes> is empty, b'\x01\x01\x01' will be use as default
"""

import socket
import sys

HOST, PORT = "localhost", 5006
data = sys.argv[1:]

# Create a socket (SOCK_STREAM means a TCP socket)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    req = bytes.fromhex("00010001000000010202" if not data else data)
    print(f"Sent:     {req}   len:{len(req)}")
    sock.sendall(req)
    # Receive data from the server and shut down
    received = sock.recv(1024)
    print(f"Received: {received}, {len(received)}")
