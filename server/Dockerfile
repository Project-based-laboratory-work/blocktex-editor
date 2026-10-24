FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-lang-japanese \
        texlive-latex-extra \
        texlive-plain-generic \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src ./src
COPY package.json ./

USER node
CMD ["node", "src/index.ts"]