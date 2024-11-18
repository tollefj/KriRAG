# setup

## docker llm server:

`make docker`

## local llm:

```bash
git clone git@github.com:ggerganov/llama.cpp.git

```

```bash
docker build -f docker/krirag-onnx-multiplatform.dockerfile -t krirag:latest .
docker save -o krirag.tar krirag:latest
docker load -i krirag.tar
```

# Notater

```
Det du kan nevne for utvikleren er at vi veldig gjerne kjører en reverse proxy foran containeren, og da bruker vi primært caddy (Web server). så veldig greit om ikke vedkommende hardkoder http:// i urler osv, men har støtte for både med og uten ssl/https

vi hadde en et prosjekt fra samarbeidene land hvor alt bortsett fra en url funka med https, fordi det var hardkodet i html-koden i frontend



```
