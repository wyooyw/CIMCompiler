class TestHelper:
    def get_image(self):
        """
        a = Buffer(<4>, int8, global);
        b = Buffer(<4>, int8, global);
        """
        import numpy as np
        self.a = np.arange(0,4,dtype=np.int8)
        self.b = np.arange(4,8,dtype=np.int8)
        image = bytearray(self.a) + bytearray(self.b)
        return image
    
    def check_image(self, image):
        """
        image should have:
        a = Buffer(<4>, int8, global); ([4,5,6,7])
        b = Buffer(<4>, int8, global); ([0,1,2,3])
        """
        import numpy as np
        new_a = np.frombuffer(image[0:4], dtype=np.int8)
        new_b = np.frombuffer(image[4:8], dtype=np.int8)
        assert np.array_equal(self.a, new_b), f"{self.a=}, {new_b=}"
        assert np.array_equal(self.b, new_a), f"{self.b=}, {new_a=}"