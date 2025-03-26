# 🚀 Project Name
## IMPORTANT ###
Before cloning this repo please run the below command
git lfs install  

#### To Run/Debug #####################
## IMPORTANT ###
Before cloning this repo please run the below command
git lfs install  


Install Python on the system

Create Virtual environment: Go to View -> Command Palette -> select Python:Create Environment -> select requirement.txt 

Goto code folder level and open command prompt and run ".\.venv\Scripts\activate" to activate the virtual environment

Go to View -> Command Palette -> select Python: Select Interpreter and choose the ('.venv') interpretter

Install dependencies: In same command prompt, go to code folder path ((.venv) C:\Users\<<UpdatePath>>\gaied-mail-swifters\code>) and run this command: pip install -r ./requirements.txt

Update "FLASK_APP" in launch.json under .vscode folder (create if not exists)
"FLASK_APP": "C:\\Users\\<absolute path>\\gaied-mail-swifters\\code\\src\\main-custom-model.py", // path to your main-custom-model.py

Open main-custom-model.py and debug using the Flask debugger (Run -> Start Debugging)

After running API, to run the UI please launch home.html (code\src\home.html) in broswer.

#####

Please refer execution demo video at artifacts\demo\Hackathon_Execution_Recording.mp4