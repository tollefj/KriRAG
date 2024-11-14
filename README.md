# Docker setup

1. download the correct llama-cpp docker image: `python docker/download-llamacpp.py`. This saves it as a tar file, for offline sharing.
2. run the llama-cpp container by `./docker/run-llama.cpp.sh`. This will host it using the current ip.

# Notater

```
Det du kan nevne for utvikleren er at vi veldig gjerne kjører en reverse proxy foran containeren, og da bruker vi primært caddy (Web server). så veldig greit om ikke vedkommende hardkoder http:// i urler osv, men har støtte for både med og uten ssl/https

vi hadde en et prosjekt fra samarbeidene land hvor alt bortsett fra en url funka med https, fordi det var hardkodet i html-koden i frontend



```
