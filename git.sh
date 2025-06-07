#!/bin/bash
# Show current git status
echo "Current git status:"
git status --untracked-files=no

# Ask whether to add all changes
#read -p "Do you want to add all changes? (Y/n): " add_response
#if [[ "$add_response" =~ ^([yY]|)$ ]]; then
#    git add -A
#else
#    echo "Exiting without committing."
#    exit 1
#fi

# Show status after staging changes
#echo "Staged changes:"
#git status

# Prompt for commit message
read -p "Enter commit message: " commit_msg
if [ -z "$commit_msg" ]; then
    echo "Commit message cannot be empty. Exiting."
    exit 1
fi

# Commit the changes
git commit -m "$commit_msg"
if [ $? -ne 0 ]; then
    echo "Commit failed. Exiting."
    exit 1
fi

# Push changes to the remote repository
git push
if [ $? -eq 0 ]; then
    echo "Push successful."
else
    echo "Push failed. Please check your remote settings."
fi
