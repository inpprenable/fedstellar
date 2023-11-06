#!/bin/bash

# Print commands and their arguments as they are executed (debugging)
set -x

# Print in console debug messages
echo "Starting services..."

# Iniciar Nginx en primer plano en un subshell para que el script continúe
nginx &

# Change directory to where app.py is located
cd /fedstellar/fedstellar/frontend

# Iniciar Gunicorn
gunicorn --worker-class eventlet --workers 1 --bind unix:/tmp/fedstellar.sock --access-logfile $SERVER_LOG app:app &

# Iniciar TensorBoard
tensorboard --host 0.0.0.0 --port 8080 --logdir $FEDSTELLAR_LOGS_DIR --window_title "Fedstellar Statistics" --reload_interval 30 --max_reload_threads 10 --reload_multifile true &

tail -f /dev/null
