tf_serve:
	tensorflow_model_server --port=8700 --rest_api_port=8701 --model_config_file=/home/doan.bao.linh/Desktop/Project/Flower/flowers102_retrieval_streamlit/flower_retrieval.config

check:
	saved_model_cli show --dir /home/doan.bao.linh/Desktop/Project/Flower/flowers102_retrieval_streamlit/temp_models/serving/1 --tag_set serve --signature_def classifier
