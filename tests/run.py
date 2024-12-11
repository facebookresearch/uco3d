import os
import unittest


if __name__ == "__main__":
    curdir = os.path.dirname(os.path.realpath(__file__))
    if True:  # run a specific test
        print("!!! REMOVE THIS !!!")
        suite = unittest.TestLoader().loadTestsFromName(
            # "test_gaussians_pcl.TestGaussiansPCL.test_visualize_gaussian_render"
            # "test_gaussians_pcl.TestGaussiansPCL.test_visualize_pcl_reprojection"
            "test_dataloader.TestDataloader.test_iterate_dataset"
            # "test_mask_depth.TestMaskDepth.test_render_unprojected_depth"
            # "test_cameras.TestCameras.test_pytorch3d_random_camera_compatibility"
            # "test_cameras.TestCameras.test_pytorch3d_real_data_camera_compatibility"
        )
        unittest.TextTestRunner().run(suite)
    else:  # run the whole suite
        suite = unittest.TestLoader().discover(curdir, pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
