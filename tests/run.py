import os
import unittest


if __name__ == "__main__":
    curdir = os.path.dirname(os.path.realpath(__file__))
    if True:  # run a specific test
        suite = unittest.TestLoader().loadTestsFromName(
            # "test_gaussians_pcl.TestGaussiansPCL.test_visualize_gaussian_render"
            "test_gaussians_pcl.TestGaussiansPCL.test_visualize_pcl_reprojection"
        )
        unittest.TextTestRunner().run(suite)
    else:  # run the whole suite
        suite = unittest.TestLoader().discover(curdir, pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
