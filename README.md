### Introduction

This file contains instructions on how to setup your Python environment in order to be able to train and test a reinforcement learning agent on a modified version of the Unity game Banana. A description of the algorithm and results can be found [here](https://github.com/ngs90/RLBanana/blob/master/Report.md)

### Set up your Python environment: 
* Install [Python 3.6.7](https://www.python.org/downloads/release/python-367/)
* Navigate to your project folder (or create one if you have not already)
    * Alternatively clone this repository directly from github: 
        * ... [gitub command] ...
* Create virtual environment in the project folder running (in PowerShell/Windows based commands):  
    * `virtualenv .env`
        * If you have multiple versions of Python installed you might need to specify which Python version virtualenv should generate the environment based on, this can be obtained with the "-p" argument. So the environment can then be created with "virtualenv -p "[PATH_TO_PYTHON_3_6_7]\python.exe" .env"
* Activate the environment 
    * `.\.env\Scripts\activate`
* Check the Python version
    * `python --version`
        * It should be 3.6.7
* Install required packages (notice we install the cpu versions of the various required learning frameworks. Can be changed to GPU but you will have to alter the requirements.txt file)
    * `pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html`

### Setup the game

The game will specify the environment that we will train our model on. The game we will be training our agent on is an altered version of the Unity environment Banana. The game is about colleccting yellow bananas (score +1 for each) and avoiding blue (score of -1 for each) bananas in a 3d world that can be navigated by moving left, right, forward and backwards. The game/environment is considered solved if we can train an agent that can get an average score of 13 or more over 100 consecutive episodes. 

Download your version below and save it in your project folder: 

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Alternatively, it is possible to [install Unity manually](https://github.com/Unity-Technologies/ml-agents/tree/master/docs). 

### Run the code
   
Having prepared the Python and Unity environment we're now ready to execute the code. 

An agent can be trained by running the following line: 

* `python train.py [path banana.exe]`
	* Example: 
		`python train.py "Banana_Windows_x86_64\Banana.exe"`


An agent can be tested running the following line: 

* `python test.py [path banana.exe] [trained model weights]`
	* Example: 
		`python test.py "Banana_Windows_x86_64\Banana.exe" "results\good_model.pth"`

### Organization of the code

* `model.py`
    * contains the specifications of the neural network used to approximate the Q-table of the state-action space. In particular the input of the network should be the state of the environment and the output is the Q-value for each action.
* `agent.py`
    * Contains available agents we can train. Available agents are:
        * AgentDQN - A Deep Q Network-agent.
        * AgentPRL - An extension of the AgentDQN where prioritized replay has been implemented using sum trees.
* `train.py`
    * Trains an agent. After training the model performance along with the model weights will be saved locally.
* `test.py`
    * Test an agent based on a pre-trained model.
