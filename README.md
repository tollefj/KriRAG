# setup

## server (docker)

```bash
./docker/build.server.sh  # clones llama.cpp and builds from cuda dockerfile
./docker/run.server.sh
```

## frontend

### local installation

```bash
cd src
python -m pip install -r requirements.txt
python install.py
streamlit run ui.py
```

### docker

./docker/build.ui.sh
./docker/run.ui.sh


```bash
docker network create krirag-net
#12029ebae1b1b18d4dabc105500e33fd8e57ddddbe0dadc733e451aa5ceb470b


```

# Notater

```
Det du kan nevne for utvikleren er at vi veldig gjerne kjører en reverse proxy foran containeren, og da bruker vi primært caddy (Web server). så veldig greit om ikke vedkommende hardkoder http:// i urler osv, men har støtte for både med og uten ssl/https

vi hadde en et prosjekt fra samarbeidene land hvor alt bortsett fra en url funka med https, fordi det var hardkodet i html-koden i frontend



```
