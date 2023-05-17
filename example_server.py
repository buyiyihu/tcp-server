"""This is a tutorial demo for the TCP service framework.

Author: buyiyihu
"""

from tcp_server.server import TCPApp

# 1. First, prepare a config. You can pass in a json file or a AppConfig object.
#   Fields of AppConfig:
#      - port. Default 0, The listening port of the service
#      - multi_worker.  Default False, Whether to activate multiple wokers,
#               if True, the service wiil be able to handle multiple requests parallelly
#      -log_format. Default [%(asctime)s] %(name)s-%(levelname)s: %(message)s
#      - request_schema: A list of Fields objects or dict
#      - response_schema: A list of Fields objects or dict
#
#
#   Fields of Attributes:
#      -  length. Required, the bytes number of this field.
#      -  name. Required, the name of this field, this will be the key
#                   when dumped from dict or loaded from bytes to dict
#      -  encoding. Default "bin", there are only option of this field, "bin" or "ascii"
#      -  data_type. Default "int", there are only option of this field, "int" or "str".
#                  When the encode is "bin", this one has to be "int", meaning to convert the byte value directly to a int.
#                  When the encode is "ascii", this one have to be "str", meaning to decode the bytes with ascii
#      -  is_identifier. Default False, indicates whether this field is the identifier of the function
#      -  is_same_as_request. Default False, indicates the value of this field will reuse the value from request,
#                   used for fixed fields like request no.
#      -  is_payload. Default False, indicates whether this field is the payload of the request/response.
#                   There must be only one payload field in the request/response.
#      -  is_status. Default False, indicates whether this field is the status field of this response.
#                   You can use no status field in your response, but if you do, remember int 0 is reserved for ok
#
#   Fields of Payload:
#      -  length. Required, the bytes number of this field.
#      -  name. Required, the name of this field, this will be the key
#                   when dumped from dict or loaded from bytes to dict
#      -  encoding. Default "bin", there are only 2 options of this field, "bin" or "ascii"
#      -  data_type. Default 'int, there are only 2 options of this field, "int" or "str".
#                  When the encode is "bin", this one has to be "int", meaning to convert the byte value directly to a int.
#                  When the encode is "ascii", this one have to be "str", meaning to decode the bytes with ascii
#
#       This is exactly the same as Fields. In fact, it works as a part of the Attributes. Several Payloads as a list make up the payload


# There 2 ways to create a config, pass a json file name or a AppConfig object
config_file = "demo_config.json"
app = TCPApp(config=config_file)


# 2. Use decorator the register functions ,
# id is identifier of the function, and cannot be duplicated.
# The parameters of the function (x as here), have to conform with the request objects
@app.function(id=1)
def test(*, x):
    # Do something
    return {"y": x + 1000}


# 3. Use decorator to the register error handlers,
# exc is type of exception occured,
# code is the representation of this kind of error, could be int or str
# This function should take a parameter e, i.e. the exception object
# The output of this function should be a 2-element tuple,
# the first one acts as error code, it will overwrite the code in the decorator,
# if you want use the one passed into decorator, use None here instead
# the second one acts as error message, it will be returned in the payload.
@app.error_handler(exc=TypeError, code=1)
def handle(e):
    # Do something
    return 16, "type error"


# After the above are set, now we can start the server.
if __name__ == "__main__":
    app.run()
