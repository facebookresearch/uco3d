# TODO: since we do not use Implicitron, just gather all low-level types in one file!
# make sure it is exactly the same file as in the exporter!

from sqlalchemy.orm import (
    composite,
    DeclarativeBase,
    Mapped,
    mapped_column,
    MappedAsDataclass,
)

from .annotation_types import (
    AlignmentAnnotation,
    CaptionAnnotation,
    DepthAnnotation,
    GaussianSplatsAnnotation,
    ImageAnnotation,
    MaskAnnotation,
    PointCloudAnnotation,
    ReconstructionQualityAnnotation,
    VideoAnnotation,
    ViewpointAnnotation,
)

# from sqlalchemy.types import TypeDecorator
from .orm_utils import TupleTypeFactory


class Base(MappedAsDataclass, DeclarativeBase):
    """subclasses will be converted to dataclasses"""


class UCO3DFrameAnnotation(Base):
    __tablename__ = "frame_annots"

    sequence_name: Mapped[str] = mapped_column(primary_key=True)
    frame_number: Mapped[int] = mapped_column(primary_key=True)
    frame_timestamp: Mapped[float] = mapped_column(index=True, nullable=True)

    image: Mapped[ImageAnnotation] = composite(
        mapped_column("_image_path"),
        mapped_column("_image_size", TupleTypeFactory(int)),
    )

    depth: Mapped[DepthAnnotation] = composite(
        mapped_column("_depth_path", nullable=True),
        mapped_column("_depth_scale_adjustment", nullable=True),
        mapped_column("_depth_mask_path", nullable=True),
    )

    mask: Mapped[MaskAnnotation] = composite(
        mapped_column("_mask_path", nullable=True),
        mapped_column("_mask_mass", index=True, nullable=True),
        mapped_column(
            "_mask_bounding_box_xywh",
            TupleTypeFactory(float, shape=(4,)),
            nullable=True,
        ),
    )

    viewpoint: Mapped[ViewpointAnnotation] = composite(
        mapped_column(
            "_viewpoint_R", TupleTypeFactory(float, shape=(3, 3)), nullable=True
        ),
        mapped_column(
            "_viewpoint_T", TupleTypeFactory(float, shape=(3,)), nullable=True
        ),
        mapped_column(
            "_viewpoint_focal_length", TupleTypeFactory(float), nullable=True
        ),
        mapped_column(
            "_viewpoint_principal_point", TupleTypeFactory(float), nullable=True
        ),
        mapped_column(
            "_viewpoint_colmap_distortion_coeffs",
            TupleTypeFactory(float, shape=(12,)),
            nullable=True,
        ),
        mapped_column("_viewpoint_intrinsics_format", nullable=True),
    )


TF3 = tuple[float, float, float]


class UCO3DSequenceAnnotation(Base):
    __tablename__ = "sequence_annots"

    sequence_name: Mapped[str] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(index=True, nullable=True)
    super_category: Mapped[str] = mapped_column(index=True, nullable=True)
    video: Mapped[VideoAnnotation] = composite(
        mapped_column("_video_path", nullable=True),
        mapped_column("_video_length", nullable=True),
    )
    mask_video: Mapped[VideoAnnotation] = composite(
        mapped_column("_mask_video_path", nullable=True),
        mapped_column("_mask_video_length", nullable=True),
    )
    depth_video: Mapped[VideoAnnotation] = composite(
        mapped_column("_depth_video_path", nullable=True),
        mapped_column("_depth_video_length", nullable=True),
    )
    sparse_point_cloud: Mapped[PointCloudAnnotation] = composite(
        mapped_column("_sparse_point_cloud_path", nullable=True),
        mapped_column("_sparse_point_cloud_n_points", nullable=True),
    )
    point_cloud: Mapped[PointCloudAnnotation] = composite(
        mapped_column("_point_cloud_path", nullable=True),
        mapped_column("_point_cloud_n_points", nullable=True),
    )
    segmented_point_cloud: Mapped[PointCloudAnnotation] = composite(
        mapped_column("_segmented_point_cloud_path", nullable=True),
        mapped_column("_segmented_point_cloud_n_points", nullable=True),
    )
    gaussian_splats: Mapped[GaussianSplatsAnnotation] = composite(
        mapped_column("_gaussian_splats_dir", nullable=True),
        mapped_column("_gaussian_splats_n_gaussians", nullable=True),
    )

    reconstruction_quality: Mapped[ReconstructionQualityAnnotation] = composite(
        mapped_column("_reconstruction_quality_viewpoint", nullable=True),
        mapped_column("_reconstruction_quality_gaussian_splats", nullable=True),
        mapped_column("_reconstruction_quality_gaussian_splats_psnr", nullable=True),
        mapped_column("_reconstruction_quality_gaussian_splats_ssim", nullable=True),
        mapped_column("_reconstruction_quality_gaussian_splats_lpips", nullable=True),
        mapped_column(
            "_reconstruction_quality_sfm_n_registered_cameras", nullable=True
        ),
        mapped_column("_reconstruction_quality_sfm_mean_track_length", nullable=True),
        mapped_column(
            "_reconstruction_quality_sfm_bundle_adjustment_final_cost", nullable=True
        ),
    )

    # captions
    caption: Mapped[CaptionAnnotation] = composite(
        mapped_column("_caption_text", nullable=True),
        mapped_column("_caption_clip_score", nullable=True),
    )

    short_caption: Mapped[CaptionAnnotation] = composite(
        mapped_column("_short_caption_text", nullable=True),
        mapped_column("_short_caption_clip_score", nullable=True),
    )

    # In right-multiply (PyTorch3D) format. X_transformed = scale * (X @ R + T)
    alignment: Mapped[AlignmentAnnotation] = composite(
        mapped_column(
            "_alignment_R", TupleTypeFactory(float, shape=(3, 3)), nullable=True
        ),
        mapped_column(
            "_alignment_T", TupleTypeFactory(float, shape=(3,)), nullable=True
        ),
        mapped_column("_alignment_scale"),
    )
