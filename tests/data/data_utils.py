import itertools
import unittest

from medsegpy.data.data_utils import compute_patches


class TestComputePatches(unittest.TestCase):
    def verify_expected_patches(self, patches, expected):
        """Verify patches are in expected patches based on """
        pass

    def test_simple(self):
        image_size = (10, 20, 30)

        # Patch size is whole volume.
        patches = compute_patches(image_size, image_size)
        assert len(patches) == 1
        patch, pad = patches[0]
        assert patch == (slice(0, 10), slice(0, 20), slice(0, 30))
        assert pad is None

        # Patch size is uniformly 1/2 on all dimensions.
        patch_size = (5, 10, 15)
        patches = compute_patches(image_size, patch_size)
        assert len(patches) == 2 * 2 * 2
        expected_patches = itertools.product(
            (slice(0, 5), slice(5, 10)),
            (slice(0, 10), slice(10, 20)),
            (slice(0, 15), slice(15, 30)),
        )
        assert all(patch in expected_patches for patch, _ in patches)
        assert all(pad is None for _, pad in patches)

        # Patch size is different on all dimensions.
        patch_size = (2, 10, 5)
        patches = compute_patches(image_size, patch_size)
        assert len(patches) == 5 * 2 * 6
        expected_patches = itertools.product(
            (slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8), slice(8, 10)),
            (slice(0, 10), slice(10, 20)),
            (
                slice(0, 5),
                slice(5, 10),
                slice(10, 15),
                slice(15, 20),
                slice(20, 25),
                slice(25, 30),
            ),  # noqa
        )
        assert all(patch in expected_patches for patch, _ in patches)
        assert all(pad is None for _, pad in patches)

    def test_padding(self):
        image_size = (10, 20, 30)
        patch_size = (5, 10, 15)

        # Pad with all dimensions with padding 1.
        patches = compute_patches(image_size, patch_size, pad_size=1)
        expected_patches = itertools.product(
            (slice(0, 4), slice(4, 9)), (slice(0, 9), slice(9, 19)), (slice(0, 14), slice(14, 29))
        )
        assert all(patch in expected_patches for patch, _ in patches)
        expected_pads = itertools.product(((1, 0), (0, 0)), ((1, 0), (0, 0)), ((1, 0), (0, 0)))
        expected_pads = tuple(None if all(px == (0, 0) for px in p) else p for p in expected_pads)
        assert tuple(pad for _, pad in patches) == expected_pads

        # Pad only first dimension
        patches = compute_patches(image_size, patch_size, pad_size=(1, None, None))
        expected_patches = itertools.product(
            (slice(0, 4), slice(4, 9)), (slice(0, 10), slice(10, 20)), (slice(0, 15), slice(15, 30))
        )
        assert all(patch in expected_patches for patch, _ in patches)
        expected_pads = itertools.product(((1, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 0)))
        expected_pads = tuple(None if all(px == (0, 0) for px in p) else p for p in expected_pads)
        assert tuple(pad for _, pad in patches) == expected_pads

        # Pad such that dimensions have padding on both front and end
        patches = compute_patches(image_size, patch_size, pad_size=(3, 5, 8))
        expected_patches = itertools.product(
            (slice(0, 2), slice(2, 7), slice(7, 10)),
            (slice(0, 5), slice(5, 15), slice(15, 20)),
            (slice(0, 7), slice(7, 22), slice(22, 30)),
        )
        assert all(patch in expected_patches for patch, _ in patches)
        expected_pads = itertools.product(
            ((3, 0), (0, 0), (0, 2)), ((5, 0), (0, 0), (0, 5)), ((8, 0), (0, 0), (0, 7))
        )
        expected_pads = tuple(None if all(px == (0, 0) for px in p) else p for p in expected_pads)
        assert tuple(pad for _, pad in patches) == expected_pads

    def test_stride(self):
        image_size = (10, 20, 30)
        patch_size = (5, 10, 15)

        # Variable strides without padding.
        patches = compute_patches(image_size, patch_size, strides=(2, 4, 6))
        expected_patches = itertools.product(
            (slice(0, 5), slice(2, 7), slice(4, 9)),
            (slice(0, 10), slice(4, 14), slice(8, 18)),
            (slice(0, 15), slice(6, 21), slice(12, 27)),
        )
        assert all(patch in expected_patches for patch, _ in patches)
        assert all(pad is None for _, pad in patches)

        # Variable strides with padding.
        patches = compute_patches(image_size, patch_size, pad_size=(3, 5, 8), strides=(2, 4, 6))
        expected_patches = list(
            itertools.product(
                (
                    slice(0, 2),
                    slice(0, 4),
                    slice(1, 6),
                    slice(3, 8),
                    slice(5, 10),
                    slice(7, 10),
                ),  # noqa
                (
                    slice(0, 5),
                    slice(0, 9),
                    slice(3, 13),
                    slice(7, 17),
                    slice(11, 20),
                    slice(15, 20),
                ),  # noqa
                (
                    slice(0, 7),
                    slice(0, 13),
                    slice(4, 19),
                    slice(10, 25),
                    slice(16, 30),
                    slice(22, 30),
                ),  # noqa
            )
        )
        assert all(patch in expected_patches for patch, _ in patches)
        expected_pads = itertools.product(
            ((3, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 2)),
            ((5, 0), (1, 0), (0, 0), (0, 0), (0, 1), (0, 5)),
            ((8, 0), (2, 0), (0, 0), (0, 0), (0, 1), (0, 7)),
        )
        expected_pads = tuple(None if all(px == (0, 0) for px in p) else p for p in expected_pads)
        assert tuple(pad for _, pad in patches) == expected_pads


if __name__ == "__main__":
    unittest.main()
