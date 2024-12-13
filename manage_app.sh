#!/bin/bash

#####################################
#
# ChatChemTS Deployment Script
#
#####################################

# Error handling
set -e 
set -o errtrace
trap "echo An error occurred during the execution." ERR

USER_ID=$(id -u)
GROUP_ID=$(id -g)
export USER_ID GROUP_ID

# Check whether OPENAI_API_KEY is set in .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Error: OPENAI_API_KEY is not set in .env file."
    exit 1
fi

# Main process
function show_help() {
    echo "Usage: $0 {deploy|stop|clean|help}"
    echo "  deploy: Deploy and start the application using docker-compose"
    echo "  start:  (Re)start the application"
    echo "  stop:   Stop the application"
    echo "  clean:  Stop and remove all resources created by the application"
    echo "  help:   Show this help message"
}

function deploy() {
    echo "Deploying the application..."
    docker compose build --no-cache
    docker compose up -d
    echo "ChatChemTS is now running! Access it at http://localhost:${CHATBOT_PORT}"
}

function start() {
    echo "Starting the existing deployment..."
    docker compose start
    echo "ChatChemTS is now running! Access it at http://localhost:${CHATBOT_PORT}"
}

function stop() {
    echo "Stopping the application..."
    docker compose stop
}

function clean() {
    echo "Cleaning up resources..."
    docker compose down --volumes --remove-orphans --rmi all
}

case "$1" in
    deploy)
        deploy
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    clean)
        clean
        ;;
    help)
        show_help
        ;;
    *)
        echo "Invalid option: $1" >&2    
        show_help
        exit 1
        ;;
esac

