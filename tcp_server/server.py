"""A customized general TCP server framework.

Author: buyiyihu
"""
import json
import keyword
import logging
import socket
import traceback
from collections import namedtuple
from enum import Enum
from itertools import chain
from socketserver import BaseRequestHandler, TCPServer, ThreadingMixIn
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field, root_validator

logger = logging.getLogger()

__all__ = ["TCPApp", "TestClient"]

HEALTH_CHECKER = "ping".encode(encoding="ascii")
HEALTH_RESPONSE = "pong".encode(encoding="ascii")


class ErrorEnum(bytes, Enum):
    request_data_decode_error = bytes.fromhex("A" * 8)
    function_retrieving_error = bytes.fromhex("B" * 8)
    request_payload_decode_error = bytes.fromhex("C" * 8)
    response_payload_encode_error = bytes.fromhex("D" * 8)
    response_data_encode_error = bytes.fromhex("E" * 8)
    unknown_exception = bytes.fromhex("F" * 8)


class Payload(BaseModel):
    """This is the payload model for requests and responses, it is a part of
    the requests and responses."""

    length: int = Field(..., description="The bytes number of this field.", gt=0)

    name: str = Field(
        ..., description="The name of this field.", min_length=1, max_length=50
    )
    encoding: Literal["bin", "ascii"] = Field(
        "bin",
        description='Encoding and decoding, there are only 2 options, "bin" or "ascii"',
    )
    data_type: Literal["int", "str"] = Field(
        "int",
        description='The type of actual data of this field, there are 2 options, "int" or "str".',
    )
    description: str = Field("", description="The descroption of this field.")

    @root_validator
    def check_values(cls, values):
        _encoding, _data_type = values.get("encoding"), values.get("data_type")
        if (_encoding == "bin") ^ (_data_type == "int"):
            raise TypeError(
                "data_type 'bin' supports only int type, and 'ascii' only str"
            )
        return values


class Attributes(BaseModel):
    """This is the field model for a complete request config, it presents the
    keys that may exist in a request or response, payload is not incluede,
    since it has its own model."""

    length: int = Field(..., description="The bytes number of this field.", ge=-1)
    name: str = Field(
        ...,
        description="The name of this field,this will be the key when \
            dumped from dict or loaded from bytes to dict",
        min_length=1,
        max_length=50,
    )
    encoding: Literal["bin", "ascii"] = Field(
        "bin",
        description='Encoding and decoding, there are only 2 options, "bin" or "ascii"',
    )
    data_type: Literal["int", "str"] = Field(
        "int",
        description='The type of actual data of this field, there are 2 options,\
             "int" or "str".',
    )
    is_identifier: bool = Field(
        False, description="Whether this field is the identifier of the function"
    )
    is_same_as_request: Union[bool, str] = Field(
        False,
        description="Indicates the value of this field will replicate the value \
            from request with the same name (if True) or the field \
                with this field's value as its name ,used for fixed fields like request no.",
    )
    is_payload: bool = Field(
        False,
        description="Whether this field is the payload of the request/response. \
            There must be only one payload field in the request/response.",
    )
    is_status: bool = Field(
        False,
        description="Whether this field is the status field of this response. \
            You can use no status field in your response, but if you do, \
                remember int 0 is reserved for ok",
    )

    @root_validator
    def check_values(cls, values):
        _encoding, _data_type = values.get("encoding"), values.get("data_type")
        if (_encoding == "bin") ^ (_data_type == "int"):
            raise TypeError(
                "data_type 'bin' supports only int type, and 'ascii' only str"
            )
        return values


class AppConfig(BaseModel):
    """This is the field model for app config, it presents the keys that should
    exist in the app config."""

    port: int = Field(0, description="The listening port of the service", ge=-1)
    multi_worker: bool = Field(
        False,
        description=" Whether to activate multiple wokers, if True,\
             the service wiil be able to handle multiple requests parallelly",
    )
    log_format: str = Field(
        "[%(asctime)s] %(name)s-%(levelname)s: %(message)s",
        description="The log formate",
    )
    log_file: str = Field("server.log", description="The log file of the service")
    request_schema: List[Attributes] = Field(
        ..., description="The list of fields of request data"
    )
    response_schema: List[Attributes] = Field(
        ..., description="The list of fields of response data"
    )


Field = namedtuple(
    "Field",
    "name,ordinal,length,is_identifier,is_same_as_request,\
        is_payload,is_status,encoding,location,data_type",
)


class Meta(type):
    """This is a metaclass for establishing connection between request schema
    and response schema."""

    def __init__(self, *args, **kwargs):
        self.request = None
        self.respsonse = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwds):
        _type = kwds.get("type")
        if _type is None:
            return super().__call__(*args, **kwds)
        elif _type == "request":
            if self.request is None:
                self.request = super().__call__(*args, **kwds)
            if self.respsonse:
                self.request.response = self.respsonse
            return self.request
        elif _type == "response":
            if self.respsonse is None:
                self.respsonse = super().__call__(*args, **kwds)
            if self.request:
                self.respsonse.request = self.request
                self.request.response = self.respsonse
            return self.respsonse
        else:
            raise ValueError(f"Illegal type:{_type}")


class Schema(metaclass=Meta):
    """This is the general schema class for requests and response, including
    type and value check and most importantly converting data between bytes and
    vaue."""

    encodes = ("bin", "ascii", "utf-8")
    ascii_types = ("int", "bool", "float", "str", "list")
    BYTE_ZERO = bytes.fromhex("00")

    def __init__(self, data: List[Union[Dict, Attributes]], type="function"):
        """Initialize the data schema.

        Args:
            - data could be a list of dicts OR pydantic models
        """
        self.type = type
        self.field_count = len(data)
        self.total_bytes = 0

        # _schema is a list of namedtuples, each tuple has 7 items
        self._schema = []
        self._schema_map = {}
        self.keys = set()
        self.identifier, self.status_field, self.payload = None, None, None
        self.links = []
        self._init_schema(
            data if isinstance(data[0], dict) else [d.dict() for d in data]
        )

    def _init_schema(self, data):
        start = 0
        for n, item in enumerate(data):
            linked, n_item = self._check(item)
            _len = n_item["length"]
            if not (_len == -1 and n != self.field_count - 1):
                ll = _len if _len > 0 else 0
                end = None if n == self.field_count - 1 else start + ll
                n_item["location"] = slice(start, end)
                n_item["ordinal"] = n
                self.total_bytes += ll
                field = self._create_field(n_item, linked)
                self._schema.append(field)
                self._schema_map[n_item["name"]] = field
                start = end
            else:
                _fields = []
                _start = None
                for _n, _item in enumerate(data[:n:-1]):
                    _linked, _n_item = self._check(_item)
                    _l = _n_item["length"]
                    _end = _start
                    _start = (-end if _end is not None else 0) - _l
                    _n_item["location"] = slice(_start, _end)
                    _n_item["ordinal"] = self.field_count - _n - 1
                    self.total_bytes += _l
                    _field = self._create_field(_n_item, _linked)
                    _fields.append(_field)
                    self._schema_map[_n_item["name"]] = _field

                n_item["location"] = slice(start, _start)
                n_item["ordinal"] = n
                field = self._create_field(n_item, linked)
                self._schema.append(field)
                self._schema.extend(reversed(_fields))
                break
        if self.type == "request" and self.identifier is None:
            raise TypeError("The request and response schema must have a identifier.")

    def _create_field(self, item, linked):
        field = Field(**item)

        if field.is_identifier:
            if self.identifier is None:
                self.identifier = field
            else:
                raise ValueError("Got multiple identifier")
        if field.is_payload:
            if self.type == "function":
                raise ValueError("A schema for a function cannot have payload")
            if self.payload is None:
                self.payload = field
            else:
                raise ValueError("Got multiple payload")
        if linked:
            self.links.append((field, linked))
        if field.is_status:
            if self.status_field is None:
                self.status_field = field
            else:
                raise ValueError("Got multiple status field")
        return field

    def _check(self, _item):
        f = Attributes(**_item)
        item = f.dict()
        _len = item["length"]
        has_len = _len > 0
        if not has_len and (self.payload or not item["is_payload"]):
            raise TypeError("Only payload as last field can have a length of 0")
        name = item["name"]
        if name in self.keys:
            raise ValueError(f"Duplicated name: {name}")
        else:
            self.keys.add(name)
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(
                f'"{name}" is not a valid name, please conform it with python identifier regulations'
            )
        if item["encoding"] not in self.encodes:
            raise ValueError(f'"encoding" has to be one of {",".join(self.encodes)}')
        if item["encoding"] == "ascii" and item["data_type"] not in self.ascii_types:
            raise ValueError(f'"type" has to be one of {",".join(self.ascii_types)}')
        is_same_as_request = item["is_same_as_request"]
        if is_same_as_request is False:
            return None, item
        if self.type == "request":
            raise ValueError("Request fields cannot be linked")
        if self.request is None:
            raise RuntimeError("Response schema should be initiated after Request")
        _name = name if is_same_as_request is True else is_same_as_request
        linked = self.request._schema_map.get(_name)
        if linked is None:
            raise ValueError(f'Cannot find "{_name}" field in request schema')
        if linked.length != item["length"]:
            raise ValueError("Unequal length from linked field")
        return linked, item

    @classmethod
    def _bytes_to_int(cls, value: bytes) -> int:
        return int(value.hex(), base=16)

    @classmethod
    def _int_to_0x(cls, value: int) -> str:
        return hex(value)[2:]

    def load(self, bytes_data):
        """To load bytes data and convert it into targeted types with the
        schema."""
        result = {}
        byte_parts = {}
        byte_len = len(bytes_data)
        if self.payload:
            if byte_len < self.total_bytes:
                raise ValueError("The bytes data is not completed")
        else:
            if not self.total_bytes == byte_len:
                raise ValueError("Invalid bytes data length")

        for i in range(self.field_count):
            sch = self._schema[i]
            tmp = bytes_data[sch.location]
            if sch.is_payload:
                # For payload part
                result[sch.name] = None
                byte_parts[sch.name] = tmp
                continue
            if sch.encoding == "bin":
                value = self._bytes_to_int(tmp)
            else:
                value = str(tmp, sch.encoding)
            result[sch.name] = value
            byte_parts[sch.name] = tmp
        if not (self.type == "request" and self.response.links):
            return result, byte_parts, None
        linking = {}
        for resp, req in self.response.links:
            linking[resp.name] = byte_parts[req.name]
        return result, byte_parts, linking

    def dump(
        self, data: Dict = None, payload: bytes = None, linking: Dict[str, bytes] = None
    ):
        """To dump python dict and convert it into bytes regulated with the
        schema."""

        incoming_keys = set()
        for key in chain(
            data or {}, self.payload and {self.payload.name} or {}, linking or {}
        ):
            if key in incoming_keys:
                raise ValueError(f"Duplicate key: '{key}' in dump data")
            incoming_keys.add(key)
        if incoming_keys != self.keys:
            raise ValueError(
                f"\nUnconformity in keys, \nexpected:{','.join(self.keys)}\nGot: {','.join(incoming_keys)}"
            )

        result = []
        for part in self._schema:
            if part.is_payload:
                if payload is None:
                    raise ValueError("Nothing in payload")
                result.append(payload)
                continue
            lk = linking.get(part.name) if linking else None
            if lk is not None:
                result.append(lk)
                continue

            value = data[part.name]
            _t = value.__class__.__name__
            if _t != part.data_type:
                raise TypeError(
                    f"Invalid type for value:{value}, expect '{part.data_type}' got '{_t}'"
                )

            if part.encoding == "bin":
                if 2 ** (8 * part.length) - 1 < value:
                    raise ValueError(f'Value for "{part.name}" is out of the range')
                if not isinstance(value, int):
                    raise TypeError("Only integer type is accepted for default encoder")
                tmp = f"{self._int_to_0x(value):0>{part.length*2}}"
                val = bytes.fromhex(tmp)
            else:
                tmp = value.encode(part.encoding)
                length = len(tmp)
                if length > part.length:
                    raise ValueError(
                        f"Lenght of {part.name} exceeds configured value {part.length}"
                    )
                if part.length != -1 and length < part.length:
                    tmp = tmp.rjust(part.length, self.BYTE_ZERO)
                val = tmp
            result.append(val)

        return b"".join(result)

    @classmethod
    def default_encoder(cls, value: Union[str, int], length=0) -> bytes:
        if isinstance(value, int):
            val = bytes.fromhex(cls._int_to_0x(value))
            right_just = True
        elif isinstance(value, str):
            val = value.encode("ascii")
            right_just = False
        else:
            raise TypeError("Unsuppot type of value")
        if length:
            if length < len(val):
                raise ValueError(
                    f"The value '{value}'' is too long for length {length}"
                )
            val = (
                val.rjust(length, cls.BYTE_ZERO)
                if right_just
                else val.ljust(length, cls.BYTE_ZERO)
            )
        return val


class FrameTCPHandler(BaseRequestHandler):
    """
    This is the main request handling function.
    After the request is received, it converts the bytes data to python objects,
    find the targeted function, and pass the data in.
    After the function is completed, it converts the python objects into bytes
    and send them back.
    Exceptions are handled automaticly and specific responses will be returned:
        0xAAAAAAAA -- request data decode error
        0xBBBBBBBB -- function retrieving error
        0xCCCCCCCC -- request payload decode error
        0xDDDDDDDD -- response payload encode error
        0xEEEEEEEE -- response data encode error

    When handling function error, the functions registered at error_handlers will be
    called.

    """

    function_map = {}
    request_schema = None
    response_schema = None

    def handle(self):
        self.data = self.request.recv(1024)

        self.logger.info(f"tcp received:  0x{self.data.hex()}")
        if self.data == HEALTH_CHECKER:
            self.request.sendall(HEALTH_RESPONSE)
            return

        try:
            request_data, byte_parts, linkings = self.request_schema.load(self.data)
        except Exception:
            self.logger.info(traceback.format_exc())
            self.request.sendall(ErrorEnum.request_data_decode_error.value)
            return
        try:
            func, func_request_schema, func_response_schema = self.function_map[
                request_data[self.request_schema.identifier.name]
            ]
        except Exception:
            self.logger.info(traceback.format_exc())
            self.request.sendall(ErrorEnum.function_retrieving_error.value)
            return

        request_payload = {}
        if self.request_schema.payload is not None:
            try:
                request_payload, _, _ = func_request_schema.load(
                    byte_parts[self.request_schema.payload.name]
                )
            except Exception:
                self.logger.info(traceback.format_exc())
                self.request.sendall(ErrorEnum.request_payload_decode_error.value)
                return
        detail, code, status = None, None, {}

        try:
            result = func(**request_payload)
        except Exception as e:
            error_func, code = self.error_handlers.get(e.__class__)
            if error_func is None:
                raise
            _code, detail = error_func(e)
            code = _code if _code is not None else code

        if self.response_schema.status_field:
            status = {self.response_schema.status_field.name: 0}
        if code is None:
            try:
                response_payload = func_response_schema.dump(data=result)
            except Exception:
                self.logger.info(traceback.format_exc())
                self.request.sendall(ErrorEnum.response_payload_encode_error.value)
                return
        else:
            if status:
                status = {self.response_schema.status_field.name: code}
            response_payload = Schema.default_encoder(detail)
        try:
            response = self.response_schema.dump(
                data=status, payload=response_payload, linking=linkings
            )
        except Exception:
            self.logger.info(traceback.format_exc())
            self.request.sendall(ErrorEnum.response_data_encode_error.value)
            return
        self.logger.info(f"sending back bytes:  {response.hex()}")
        self.request.sendall(response)


class FrameTCPServerMixin:
    """Server exception error handler, return 0xFFFFFFFF when unknow exceptions
    are met."""

    def handle_error(self, request, client_address):
        self.logger.error(
            f"\nException occurs when handling a request from {client_address[0]}:{client_address[1]}"
        )
        self.logger.info(traceback.format_exc())
        request.sendall(ErrorEnum.unknown_exception.value)


class TCPApp:
    """Main server class."""

    WORDS_START = (
        "\n" + "+" * 20 + "\n* TCP Server Started\n" + "+" * 20 + "\n" + "= PORT:{port}"
    )

    def __init__(self, name="app", *, config):
        self.name = name

        if isinstance(config, str):
            with open(config) as f:
                configs = json.load(f)
        elif not isinstance(config, AppConfig):
            raise TypeError("Config must be either AppConfig object or a json file")
        else:
            configs = config.dict()

        self.initialized = False

        self.default_host = "0.0.0.0"
        self.port = configs.get("port", 0)
        _base = (
            (ThreadingMixIn, FrameTCPServerMixin, TCPServer)
            if configs.get("multi_worker")
            else (FrameTCPServerMixin, TCPServer)
        )
        self.server_class = type("Server", _base, {})

        self.request_no_max = configs.get("request_no_max", 0)
        self.order_count_volume = configs.get("order_count_volume", 0)
        self.status_code_volume = configs.get("status_code_volume", 1)
        self.request_no_max = configs.get("request_no_max")

        self.logger = logger.info("TCP framework")

        self.function_map = {}
        self.function_schema_map = {}
        self.error_handlers = {}

        self.request_function_json = self._extrct_function_info(
            configs["request_schema"]
        )
        self.response_function_json = self._extrct_function_info(
            configs["response_schema"]
        )

        self._initialize_function_schemas(self.request_function_json)
        self._initialize_function_schemas(self.response_function_json)

        self.request_schema = Schema(configs["request_schema"], type="request")
        self.response_schema = Schema(configs["response_schema"], type="response")
        self.func_id_field = self.request_schema.identifier

    def set_ip(self, ip):
        self.logger.info("ip reset to: %s", ip)
        self.default_host, self.port = ip.split(":")
        self.port = int(self.port)

    def _extrct_function_info(self, schema_list):
        for sch in schema_list:
            if not sch.get("is_payload"):
                continue
            data = sch.pop("respective_details")
        if not data:
            raise ValueError("Function schema info not claimed")
        return data

    def _initialize_function_schemas(self, data):
        for func_data in data:
            payloads = [Payload(**field) for field in func_data["details"]]
            func_schema = Schema(payloads)
            self.function_schema_map.setdefault(func_data["identifier"], []).append(
                func_schema
            )

    def function(self, id):
        """Register functions for different instructions."""

        if self.initialized:
            raise RuntimeError
        if id in self.function_map:
            raise ValueError("Duplicated function id")
        if not (
            isinstance(id, int)
            and self.func_id_field.encoding == "bin"
            or isinstance(id, str)
            and self.func_id_field.encoding == "ascii"
        ):
            raise TypeError(f"Illegal func identifier type with value {id}")

        def wrapper(func):
            self.function_map[id] = func, *self.function_schema_map[id]
            return func

        return wrapper

    def error_handler(self, *, exc, code):
        """Register functions for different types of error_handlers.

        These handlers will use any exception class as key, and a
        function as value. When the type of exception in key is raised,
        the value will be called.
        """
        if self.initialized:
            raise RuntimeError

        def wrapper(func):
            self.error_handlers[exc] = func, code
            return func

        return wrapper

    def run(self):
        """Start the server.

        Usage:
            app.run()
        """
        self.initialized = True
        FrameTCPHandler.function_map = self.function_map
        FrameTCPHandler.error_handlers = self.error_handlers
        FrameTCPHandler.request_schema = self.request_schema
        FrameTCPHandler.response_schema = self.response_schema
        FrameTCPHandler.logger = self.logger
        self.server = self.server_class(
            (self.default_host, self.port), FrameTCPHandler
        )
        self.server.logger = self.logger

        with self.server:
            _, port = self.server.server_address
            self.logger.info(self.WORDS_START.format(port=port))
            self.server.serve_forever()

    @property
    def function_ids(self):
        return list(self.function_schema_map.keys())


class TestClient:

    REQUEST_FORMAT = (
        "Sending a request >>\n\n"
        "request dict:\n\n {req_dict}\n\n"
        "request bytes:\n\n{req_bytes}\n\n"
        "------->\n\n"
        "response bytes:\n\n{res_bytes}\n\n"
        "response dict:\n\n{res_dict}\n\n"
        "[done]"
    )

    def __init__(self, *, host, config):
        self.host = host

        self.app = TCPApp(config=config)
        self.port = self.app.port
        self.request_schema = self.app.request_schema
        self.response_schema = self.app.response_schema
        self.payload_schemas = self.app.function_schema_map

        self.error_map = {e.value: e.name for e in ErrorEnum}

    def test(self, req_dict: Dict):
        result = {"req_dict": json.dumps(req_dict, indent=4)}
        req_payload_dict = {}
        func_id = req_dict[self.request_schema.identifier.name]
        request_payload_schema, response_payload_schema = self.payload_schemas[func_id]
        for key in request_payload_schema.keys:
            req_payload_dict[key] = req_dict.pop(key)

        req_payload_bytes = request_payload_schema.dump(data=req_payload_dict)
        req_bytes = self.request_schema.dump(data=req_dict, payload=req_payload_bytes)
        result["req_bytes"] = req_bytes

        res_bytes = self._request(req_bytes)
        result["res_bytes"] = res_bytes

        error = self.error_map.get(res_bytes)
        if error:
            result["res_dict"] = error
            print(self.REQUEST_FORMAT.format(**result))
            return
        response_dict, byte_parts, _ = self.response_schema.load(res_bytes)
        response_dict.pop(self.response_schema.payload.name)
        response_payload_bytes = byte_parts[self.response_schema.payload.name]
        if (
            self.response_schema.status_field
            and response_dict[self.response_schema.status_field.name] != 0
        ):
            response_dict["error_message"] = response_payload_bytes.decode()
            result["res_dict"] = json.dumps(response_dict, indent=4)
            print(self.REQUEST_FORMAT.format(**result))
            return
        response_payload_dict, _, _ = response_payload_schema.load(
            response_payload_bytes
        )
        response_dict.update(response_payload_dict)
        result["res_dict"] = json.dumps(response_dict, indent=4)
        print(self.REQUEST_FORMAT.format(**result))

    def check_health(self):
        res_bytes = self._request(HEALTH_CHECKER)
        if res_bytes == HEALTH_RESPONSE:
            print("Health check: OK")
        else:
            print(f"Health check: Not ok, received: {res_bytes.decode('ascii')}")

    def _request(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(data)
            received = sock.recv(1024)
        return received
