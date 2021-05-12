#!/usr/bin/env bash
xvfb-run -s "-screen 0 640x480x24 +extension GLX" python play_ddpg.py -m $1 -r $2
