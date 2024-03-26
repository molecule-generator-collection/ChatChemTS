#!/bin/bash

#####################################
#
# ChatChemTS Deployment Script
#
#####################################

USER_ID=$(id -u)
GROUP_ID=$(id -g)
export USER_ID GROUP_ID

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
    echo "ChatChemTS is now running! Access it at http://localhost:8000"
}

function start() {
    echo "Starting the existing deployment..."
    docker compose start
    echo "ChatChemTS is now running! Access it at http://localhost:8000"
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

