#!/bin/bash

function usage {
  echo "Usage: `basename $0` [-u] [-n <docker_container_name>] [-i <docker_image>] [-d <docker_home_dir>] [-A <custom_args>] [docker command]"
}

# handling an invalid name: replace slashs by "-"
USER_NO_SLASH=$(printf '%s\n' "$USER" | sed -e 's/[\/&]/-/g')
name=${USER_NO_SLASH}-bash-$(date +'%Y%m%d-%H%M%S.%4N')

while [[ $# -gt 0 ]]
do
    key=$1
    case $key in
        -h|--help)
        usage
        exit 0
        ;;
        -u|--user)
        user="$USER"
        shift # passed argument
        ;;
        -n|--name)
        name="$2"
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
        -A|--custom_args)
        custom_args="$2"
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


cmd="$@"

# handling an invalid name: replace slashs by double slashs
escaped_user=$(printf '%s\n' "$user" | sed -e 's/[\/&]/\\&/g')
#echo escaped_user = $escaped_user

echo "`basename $0`: POS_CMD = $cmd custom_args = $custom_args"

echo -e "\033[0;1mInitiating...\033[0;0m"
[[ ! -z $name ]] && name_arg="--name $name"
if [[ ! -z $user ]]
then
    if [[ -z $DOCKER_HOME_DIR ]]
    then
        DOCKER_HOME_DIR=$HOME
    fi
    echo "Docker home directory: $DOCKER_HOME_DIR"
    user_stuff="--user=$(id -u $user):$(id -g $user) -v $DOCKER_HOME_DIR:$HOME -e USER=$escaped_user -e HOME=$HOME -e PATH=$PATH -e PYTHONPATH=$PYTHONPATH"
else
    user_stuff=""
fi

echo -e "\033[0;1mRunning docker...\033[0;0m"
docker_bin="nvidia-docker"
[[ -z $(command -v $docker_bin) ]] && docker_bin="docker"

full_cmd="$docker_bin run -it --rm -e DISPLAY=$DISPLAY"
full_cmd="$full_cmd -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v /etc/shadow:/etc/shadow -v /etc/sudoers:/etc/sudoers -v /usr/share/bash-completion/bash_completion:/usr/share/bash-completion/bash_completion"
full_cmd="$full_cmd -v /data:/data -v /ssd3:/ssd3 -v /var/run/docker.sock:/var/run/docker.sock -v /etc/localtime:/etc/localtime:ro -v /run/dbus/:/run/dbus/ -v /dev/shm:/dev/shm"
# allow debugging with dbg under the docker
# 	https://stackoverflow.com/questions/35860527/warning-error-disabling-address-space-randomization-operation-not-permitted
#	https://docs.docker.com/engine/reference/run/
full_cmd="$full_cmd --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
full_cmd="$full_cmd $custom_args $user_stuff $name_arg $DOCKER_IMAGE $cmd"

echo "$full_cmd"
eval "$full_cmd"

