# Use an official Node.js runtime as a parent image
FROM node:22

# Set the working directory in the container
WORKDIR /app

# Copy package.json and yarn.lock
COPY package.json yarn.lock ./

# Install the required packages using Yarn
RUN yarn install

# Copy the rest of your application code into the container
COPY . .

# Build the React application
RUN yarn build

# Install serve to serve the static files
RUN yarn global add serve

# Expose the port your React app runs on
EXPOSE 3000

# Command to serve the built application
CMD ["serve", "-s", "build"]
