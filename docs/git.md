To change the Git repository's upstream URL from HTTP-based to SSH-based, you can follow these steps:

1. **Check the current remote URL**:
   Open a terminal and navigate to your repository, then run the following command to see the current remote URL:

   ```bash
   git remote -v
   ```

   This will display the current URL for `origin` (or any other remote you are using).

2. **Change the remote URL to SSH**:
   Use the following command to change the remote URL from HTTP to SSH. Replace `repository-url` with your repository's SSH URL (e.g., `git@github.com:username/repository.git`):

   ```bash
   git remote set-url origin git@github.com:username/repository.git
   ```

3. **Verify the change**:
   Run the following command again to ensure the remote URL has been updated:

   ```bash
   git remote -v
   ```

   You should now see the SSH URL for the `origin` remote.

These steps will update your repository's remote URL to use SSH, allowing for secure and authenticated connections.