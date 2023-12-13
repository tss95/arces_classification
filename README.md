## Setting Up the Environment Variable

The application uses an environment variable `ROOT_DIR` to determine the root directory of the project. You need to set this environment variable before running the application.

### On Unix/Linux/macOS

Open your terminal and enter the following command:

```bash
export ROOT_DIR=/path/to/your/root/directory
```	


Replace /path/to/your/root/directory with the actual path to your root directory.

**On Windows**
Open Command Prompt and enter the following command:

Replace C:\path\to\your\root\directory with the actual path to your root directory.

Please note that these commands will only set the ROOT_DIR environment variable for the current session. If you open a new terminal or Command Prompt window, you will need to set the environment variable again.

To set the environment variable permanently, you can add the above command to your shell's startup file (like ~/.bashrc or ~/.bash_profile on Unix/Linux/macOS, or the Environment Variables on Windows).

