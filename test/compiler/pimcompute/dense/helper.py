import numpy as np
class TestHelper:
    def set_image(self):
        """
        input = Buffer(<4>, int8, global);
        weight = Buffer(<4,4>, int8, global);
        """
        self.input = np.arange(4,dtype=np.int8)
        self.weight = np.arange(16,dtype=np.int8)
        image = bytearray(self.input) + bytearray(self.weight)
        return image
    
    def check_image(self, image):
        """
        image should have:
        input (4 byte)
        weight (16 byte)
        output (16 byte)
        """
        output = np.frombuffer(image[20:36], dtype=np.int32)
        golden = np.dot(self.input.astype(np.int32), self.weight.astype(np.int32))
        assert (output==golden).all(), f"{output=}, {golden=}"