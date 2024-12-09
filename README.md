# setup

### docker

```bash
chmod +x run-krirag.sh
.run-krirag.sh
```

### docker (server only)

```bash
./docker/build.server.sh  # clones llama.cpp and builds from cuda dockerfile
./docker/run.server.sh
```

### frontend/ui 

```bash
cd src
python -m pip install -r requirements.txt
python install.py
streamlit run ui.py
```
