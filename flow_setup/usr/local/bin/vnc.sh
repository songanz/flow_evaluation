#!/bin/bash

export RESOLUTION=$1

# Set password
if [ ! -f $HOME/.vnc/passwd ]; then
    x11vnc -storepasswd
fi

if [ -n "$X11VNC_ARGS" ]; then
    sed -i "s/^command=x11vnc.*/& ${X11VNC_ARGS}/" /etc/supervisor/conf.d/supervisord.conf
    echo "X11VNC_ARGS: $X11VNC_ARGS"
fi

if [ -n "$OPENBOX_ARGS" ]; then
    sed -i "s#^command=/usr/bin/openbox\$#& ${OPENBOX_ARGS}#" /etc/supervisor/conf.d/supervisord.conf
    echo "OPENBOX_ARGS: $OPENBOX_ARGS"
fi

# Set screen resolution
if [ -n "$RESOLUTION" ]; then
    sed -i "s/1024x768/$RESOLUTION/" /usr/local/bin/xvfb.sh
fi

USER=${USER:-root}


# escape $USER and $HOME (backslash, etc.)
# see https://stackoverflow.com/questions/407523/escape-a-string-for-a-sed-replace-pattern
ESCAPED_HOME=$(printf '%s\n' "$HOME" | sed -e 's/[\/&]/\\&/g')
ESCAPED_USER=$(printf '%s\n' "$USER" | sed -e 's/[\/&]/\\&/g')

sed -i -e "s|%USER%|$ESCAPED_USER|" -e "s|%HOME%|$ESCAPED_HOME|" /etc/supervisor/conf.d/supervisord.conf

# home folder
mkdir -p $HOME/Desktop
mkdir -p $HOME/.config

if [ ! -x "$HOME/Desktop/lxterminal" ]; then
	ln -s /usr/local/bin/lxterminal.sh $HOME/Desktop/lxterminal
fi
if [ ! -x "$HOME/.config/lxterminal" ]; then
	ln -s /etc/lxde/lxterminal $HOME/.config/lxterminal
	echo "Linking lxterminal"
fi
if [ ! -x "$HOME/.config/lxpanel" ]; then
	ln -s /etc/lxde/lxpanel $HOME/.config/lxpanel
	echo "Linking lxpanel"
fi
if [ ! -x "$HOME/.config/pcmanfm" ]; then
	ln -s /etc/lxde/pcmanfm $HOME/.config/pcmanfm
	echo "Linking pcmanfm"
fi

if [ ! -d "$HOME/.cache/" ]; then
	ln -s /etc/lxde/.cache $HOME/.cache
	echo "Linking .cache"
  if [ -d /data/$USER ]; then
      mkdir -p /data/$USER/.cache/JetBrains
      ln -s /data/$USER/.cache/JetBrains /etc/lxde/.cache/JetBrains
      echo "Linking .cache/JetBrains to /data/$USER/.cache/JetBrains"
  fi
fi

if [ ! -d "$HOME/.config/JetBrains" ]; then
  if [ -d /data/$USER ]; then
	  mkdir -p /data/$USER/.config/JetBrains
	  ln -s /data/$USER/.config/JetBrains $HOME/.config/JetBrains
	  echo "Linking .config/JetBrains to /data/$USER/.config/JetBrains"
	fi
fi

if [ ! -x "$HOME/.config/pcmanfm/LXDE/" ]; then
    mkdir -p $HOME/.config/pcmanfm/LXDE/
    ln -sf /usr/local/share/doro-lxde-wallpapers/desktop-items-0.conf $HOME/.config/pcmanfm/LXDE/
    chown -R $USER $HOME
fi

printf "\n\n"
echo "IMPORTANT: YOU ARE NOW RUNNING ON DOCKER..."
echo "Press ^p and afterwards ^q to return to server, or type \"exit\" to close this docker and the VNC server."
exec /bin/tini -- supervisord -n -c /etc/supervisor/supervisord.conf > ~/tini.log 2>&1 & /bin/bash -c "cd $HOME && /bin/bash"
