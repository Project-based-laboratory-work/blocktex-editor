FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-lang-japanese \
        texlive-latex-extra \
        texlive-plain-generic \
        latexmk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
