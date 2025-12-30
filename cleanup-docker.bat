@echo off
echo Stopping all containers...
docker stop $(docker ps -aq) 2>nul

echo Stopping docker-compose services...
docker-compose down

echo Removing all containers...
docker container prune -f

echo Removing all images...
docker image prune -a -f

echo Removing all volumes...
docker volume prune -f

echo Removing all networks...
docker network prune -f

echo Removing build cache...
docker builder prune -a -f

echo Complete system cleanup...
docker system prune -a -f --volumes

echo Docker cleanup completed!
docker system df