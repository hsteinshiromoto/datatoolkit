#!/usr/bin/env bash

# References
# [1] https://www.cyberciti.biz/faq/linux-change-user-group-uid-gid-for-all-owned-files/
# [2] https://www.thegeekdiary.com/how-to-correctly-change-the-uid-and-gid-of-a-user-group-in-linux/
# [3] https://askubuntu.com/questions/16700/how-can-i-change-my-own-user-id
# [4] https://askubuntu.com/questions/617850/changing-from-user-to-superuser

set -e

# If "-e uid={custom/local user id}" flag is not set for "docker run" command, use 9999 as default
CURRENT_UID=${uid:-9999}
CURRENT_GID=${gid:-9999}

# Modify user "vscode" with selected UID and GID
su - # [4]
usermod -u $CURRENT_UID vscode
groupmod -g $CURRENT_GID vscode

# Execute process as root
exec gosu vscode "$@"