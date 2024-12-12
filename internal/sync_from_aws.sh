###### uco3d_sample ######
# hyperloop_client \
#   -command submitJob \
#   -src_path "s3://genai-transfer/dnovotny/datasets/uco3d_sample.zip" \
#   -dest_path "manifold://coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample.zip" \
#   -src_file_system_options '{"s3_account_id":"957341995209", "s3_sso_role":"SSOS3ReadWriteGenaiTransfer", "s3_region":"us-east-1"}' \
#   -dest_file_system_options '{"manifold_api_key": "coreai_3d-key"}' \
#   -frontend_tier hyperloop.frontend.prod \
#   -pool_group genai_s3_to_mf \
#   -is_dir false
# # cd /home/dnovotny/nha-wsf/dnovotny/datasets/
# # manifold get coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample.zip
# mkdir /home/dnovotny/data/
# cd /home/dnovotny/data/
# manifold get coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample.zip
# unzip uco3d_sample.zip
# cd uco3d_sample
# ln -s ./metadata_vgg_1128_test15.sqlite ./metadata.sqlite

###### uco3d_sample_v2 ######
# hyperloop_client \
#   -command submitJob \
#   -src_path "s3://genai-transfer/dnovotny/datasets/uco3d_sample_v2/" \
#   -dest_path "manifold://coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample_v2/" \
#   -src_file_system_options '{"s3_account_id":"957341995209", "s3_sso_role":"SSOS3ReadWriteGenaiTransfer", "s3_region":"us-east-1"}' \
#   -dest_file_system_options '{"manifold_api_key": "coreai_3d-key"}' \
#   -frontend_tier hyperloop.frontend.prod \
#   -pool_group genai_s3_to_mf \
#   -is_dir true
# mkdir -p /home/dnovotny/nha-wsf/dnovotny/datasets/uco3d_sample_v2/
# manifold getr \
#   coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample_v2/ \
#   /home/dnovotny/data/datasets/
# hyperloop_client -command getJobReport -job_id prod.pci:47:2552888016:genai_s3_to_mf


###### uco3d mega ######
# hyperloop_client \
#   -command submitJob \
#   -src_path "s3://genai-transfer/uco3d/1211/dataset_export/" \
#   -dest_path "manifold://coreai_3d/tree/fsx-repligen/shared/datasets/uCO3D/dataset_export_241211/" \
#   -src_file_system_options '{"s3_account_id":"957341995209", "s3_sso_role":"SSOS3ReadWriteGenaiTransfer", "s3_region":"us-east-1"}' \
#   -dest_file_system_options '{"manifold_api_key": "coreai_3d-key"}' \
#   -frontend_tier hyperloop.frontend.prod \
#   -pool_group genai_s3_to_mf \
#   -is_dir true
hyperloop_client -command getJobReport -job_id prod.pci:113:2443572145:genai_s3_to_mf
