#!/bin/bash

#####################################
#
# ChatChemTS Deployment Script
#
#####################################

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
    docker compose up -d
}

function start() {
    echo "Starting the existing deployment..."
    docker compose start
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
