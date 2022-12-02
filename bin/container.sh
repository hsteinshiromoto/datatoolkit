#!/usr/bin/env bash

# ---
# Functions
# ---

# Documentation
display_help() {
    echo "Usage: [variable=value] $0" >&2
    echo
    echo "   -h, --help                 display help"
    echo "   -p, --pull                 pull container"
    echo
    # echo some stuff here for the -a or --add-options
    exit 1
}

make_variables() {
    PROJECT_NAME=$(basename $(git remote get-url origin) | sed 's/\.[^.]*$//')
    LATEST_VERSION=$(git tag -l --sort=-creatordate | head -n 1 | cut -d "v" -f2-)
    LATEST_VERSION="latest"

    IMAGE="ghcr.io/hsteinshiromoto/${PROJECT_NAME}/${PROJECT_NAME}:${LATEST_VERSION}"
    LOCAL_IMAGE="hsteinshiromoto/${PROJECT_NAME}:${LATEST_VERSION}"
}

pull_container() {
    make_variables

    docker pull ${IMAGE}
    docker tag ${IMAGE} ${LOCAL_IMAGE}
}

# Available options
while :
do
    case "$1" in
        -h | --help)
            display_help
            exit 0
            ;;
        
        -p | --pull_container)
            pull_container
            exit 0
            ;;

        "")
            display_help  # Call your function
            break
            ;;

        --) # End of all options
            shift
            break
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            ## or call function display_help
            exit 1
            ;;

      *)  # No more options
            break
            ;;
    esac
done



