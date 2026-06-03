FROM node:20-alpine

RUN apk --no-cache add curl
WORKDIR /app

COPY --chown=1000:1000 package*.json ./
RUN npm ci

COPY --chown=1000:1000 . .
RUN npm run build
EXPOSE 3000
USER 1000

CMD ["npm", "run", "serve"]
