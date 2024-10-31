class TestHelper:
    def get_image(self):
        import numpy as np

        """
        input = Buffer(<2,4>, int8, global);
        weight = Buffer(<2,4,8>, int8, global);
        """
        self.input = np.arange(8, dtype=np.int8).reshape(2, 4)
        self.weight = np.arange(64, dtype=np.int8).reshape(2, 4, 8)
        image = bytearray(self.input) + bytearray(self.weight)
        return image

    def check_image(self, image):
        import numpy as np

        """
        image should have:
        input (8 * 1 byte)
        weight (64 * 1 byte)
        output (8 * 4 byte)
        """
        output = np.frombuffer(image[72:104], dtype=np.int32)
        golden_g0 = np.dot(
            self.input[0, :].astype(np.int32), self.weight[0, :, :].astype(np.int32)
        )
        golden_g1 = np.dot(
            self.input[1, :].astype(np.int32), self.weight[1, :, :].astype(np.int32)
        )
        golden = golden_g0 + golden_g1
        assert np.array_equal(output, golden), f"{output=}, {golden=}"
