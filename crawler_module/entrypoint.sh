#!/bin/bash

function shutdown {
  kill -s SIGTERM $XVFB_PID
  wait $XVFB_PID
}

export DISPLAY=:0

Xvfb $DISPLAY -screen 0 1024x768x24 -ac &
XVFB_PID=$!

trap shutdown SIGTERM SIGINT
for i in $(seq 1 10)
do
  xdpyinfo -display $DISPLAY >/dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "vnc server started"
    break
  fi
  echo Waiting for xvfb...
  sleep 0.5
done

ratpoison &
x11vnc -display $DISPLAY -bg -nopw -xkb -shared -repeat -loop -forever &

python app.py

wait $XVFB_PID
