class TestHelper:
    def get_image(self):
        import numpy as np

        """
        input = Buffer(<2,4>, int8, global);
        weight = Buffer(<4,2,4>, int8, global);
        """
        self.input = np.arange(8, dtype=np.int8).reshape(2, 4)
        self.weight = np.arange(32, dtype=np.int8).reshape(4, 2, 4)
        image = bytearray(self.input) + bytearray(self.weight)
        return image

    def check_image(self, image):
        import numpy as np

        """
        image should have:
        input (8 * 1 byte)
        weight (32 * 1 byte)
        output (8 * 4 byte)
        """
        output = np.frombuffer(image[40:72], dtype=np.int32)
        golden_g0 = np.dot(
            self.input[0, :].astype(np.int32), self.weight[:, 0, :].astype(np.int32)
        )
        golden_g1 = np.dot(
            self.input[1, :].astype(np.int32), self.weight[:, 1, :].astype(np.int32)
        )
        golden = np.concatenate([golden_g0, golden_g1])
        assert np.array_equal(output, golden), f"{output=}, {golden=}"
