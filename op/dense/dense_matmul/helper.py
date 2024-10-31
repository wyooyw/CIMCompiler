class TestHelper:
    """
    C = AB
    A: [8,8] int8
    B: [8,8] int8, macro
    C: [8,8] int32
    """

    def get_image(self):
        import numpy as np

        """
        input = Buffer(<8,8>, int8, global);
        weight = Buffer(<8,8>, int8, global);
        """
        self.input = np.arange(64, dtype=np.int8).reshape(8, 8)
        self.weight = np.arange(64, dtype=np.int8).reshape(8, 8)
        image = bytearray(self.input) + bytearray(self.weight)
        return image

    def check_image(self, image):
        import numpy as np

        """
        image should have:
        input (64 * 1 byte)
        weight (64 * 1 byte)
        output (64 * 4 byte)
        """
        output = np.frombuffer(image[128:384], dtype=np.int32).reshape(8, 8)
        golden = np.matmul(self.input.astype(np.int32), self.weight.astype(np.int32))
        print("golden:", golden)
        print("output:", output)
        assert np.array_equal(output, golden), f"{output=}, {golden=}"
