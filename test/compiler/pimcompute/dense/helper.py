class TestHelper:
    def get_image(self):
        import numpy as np
        """
        input = Buffer(<4>, int8, global);
        weight = Buffer(<4,8>, int8, global);
        """
        self.input = np.arange(4,dtype=np.int8)
        self.weight = np.arange(32,dtype=np.int8).reshape(4,8)
        image = bytearray(self.input) + bytearray(self.weight)
        return image
    
    def check_image(self, image):
        import numpy as np
        """
        image should have:
        input (4 * 1 byte)
        weight (32 * 1 byte)
        output (8 * 4 byte)
        """
        output = np.frombuffer(image[36:68], dtype=np.int32)
        golden = np.dot(self.input.astype(np.int32), self.weight.astype(np.int32))
        assert np.array_equal(output,golden), f"{output=}, {golden=}"