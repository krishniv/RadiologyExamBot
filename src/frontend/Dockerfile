FROM node:14

WORKDIR /src

COPY package.json package-lock.json ./

RUN yarn install

COPY . .

EXPOSE 3000

# Command to run the Node.js app
CMD ["yarn", "start"]
