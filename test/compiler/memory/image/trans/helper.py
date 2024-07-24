class TestHelper:
    def get_image(self):
        """
        buf = Buffer(<4>, int8, global);
        """
        import numpy as np
        self.input = np.arange(4,dtype=np.int8)
        image = bytearray(self.input)
        return image
    
    def check_image(self, image):
        """
        image should have:
        buf (4 byte)
        buf_copy (4 byte)
        """
        import numpy as np
        output = np.frombuffer(image[4:8], dtype=np.int8)
        assert output.shape==self.input.shape, "shape not match"
        assert (output==self.input).all(), f"{output=}, {self.input=}"