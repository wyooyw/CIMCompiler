import base64
def parse_multi_image(json_obj):
    for key, value in json_obj.items():
        data = base64.b64decode(image)
        data = bytearray(data)