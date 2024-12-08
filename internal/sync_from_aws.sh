hyperloop_client \
  -command submitJob \
  -src_path "s3://genai-transfer/dnovotny/datasets/uco3d_sample.zip" \
  -dest_path "manifold://coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample.zip" \
  -src_file_system_options '{"s3_account_id":"957341995209", "s3_sso_role":"SSOS3ReadWriteGenaiTransfer", "s3_region":"us-east-1"}' \
  -dest_file_system_options '{"manifold_api_key": "coreai_3d-key"}' \
  -frontend_tier hyperloop.frontend.prod \
  -pool_group genai_s3_to_mf \
  -is_dir false

# cd /home/dnovotny/nha-wsf/dnovotny/datasets/
# manifold get coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample.zip
mkdir /home/dnovotny/data/
cd /home/dnovotny/data/
manifold get coreai_3d/tree/fsx-repligen/dnovotny/datasets/uco3d_sample.zip
unzip uco3d_sample.zip
