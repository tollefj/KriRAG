default: install run
install:
	pip install -r requirements.txt
	python3 src/install.py
run:
	cd src && streamlit run ui.py

# install-gpu:
# run docker with
# --gpus all
# after installing nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

docker:
	chmod +x docker/01-download-llama.cpp.sh
	chmod +x docker/02-run-llama.cpp.sh
	./docker/01-download-llama.cpp.sh
	./docker/02-run-llama.cpp.sh
