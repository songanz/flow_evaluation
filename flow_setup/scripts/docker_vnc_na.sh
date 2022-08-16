#!/bin/bash

function usage {
  echo "Usage: `basename $0` -p <port> [-g WxH] [-i <docker_image>] [-d <docker_home_dir>] [-k desktop] [arguments to pass to 'docker run']"
}

POSITIONAL=()

port=""
geometry=1920x1080
desktop=""

while [[ $# -gt 0 ]]
do
    key=$1
    case $key in
        -h|--help)
        usage
        exit 0
        ;;
        -p|--port)
        port="$2"
        shift # passed value
        shift # passed value
        ;;
        -g|--geometry)
        geometry="$2"
        shift # passed value
        shift # passed value
        ;;
        -i|--image)
        DOCKER_IMAGE="$2"
        shift # passed value
        shift # passed value
        ;;
        -d|--docker_home)
        DOCKER_HOME_DIR="$2"
        shift # passed value
        shift # passed value
        ;;
        -k|--desktop)
        desktop="$2"
        shift # passed value
        shift # passed value
        ;;
      --)
	    shift # passed argument
        break # end
	    ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # passed argument
        ;;
    esac
done

###########################################
# Remaining positional arguments after -- #
###########################################
while [[ $# -gt 0 ]]
do
	POSITIONAL+=("$1") # save it in an array for later
	shift # passed argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [[ -z $port ]]
then
    usage
    exit 1
fi

# test invalid username
#USER="SOMEDOMAIN\someuser"

# handling an invalid name: replace slashs by "-"
USER_NO_SLASH=$(printf '%s\n' "$USER" | sed -e 's/[\/&]/-/g')
#echo USER_NO_SLASH = $USER_NO_SLASH

# additional arguments to pass to 'docker run', e.g. "-v /my_dir:/my_dir"
POS_ARGS="$@"
echo "`basename $0`: POS_ARGS = $POS_ARGS"

port=$((5900 + $port))
name=${USER_NO_SLASH}-VNC-${port}
#echo name = $name
cmd="/usr/local/bin/vnc.sh $geometry 2560x1350 1920x1080 1920x980 1600x1110 1024x768 3840x1080 3840x980"

arg_image=""
if [[ ! -z $DOCKER_IMAGE ]]
then
    arg_image="-i $DOCKER_IMAGE"
fi

arg_home=""
if [[ ! -z $DOCKER_HOME_DIR ]]
then
    arg_home="-d $DOCKER_HOME_DIR"
fi

`dirname $0`/docker_bash.sh -A "-p $port:5900 -e MY_DESKTOP=$desktop $POS_ARGS" -u -n $name $arg_image $arg_home $cmd

