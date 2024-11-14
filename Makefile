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
