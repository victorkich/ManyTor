# User Guide
To train and test this project is pretty simple. With only two lines of code you complete this task. 
First, you need the clone this repository. For do this use:

`git clone https://github.com/victorkich/ReinforcementLearning-IK`

Now, you need chose what algorithm do you want run, PPO or DDPG. 
If you want run the DDPG execute the following code:

`python3 ddpg.py --test`

And for the PPO this:

`python3 ppo.py --test`

_Be aware you have tensorflow v2 installed. Even as numpy, pandas, tqdm, keras and matplotlib._

# The Project

In this project i chose to train and test 2 great algorithms to solve the continuous inverse kinematics with multiple objectives.
These algorithms is Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG).

<h2>The Environment</h2>
For general purpose, i decided to create a new environmnet instead use one environment already created.
Is this project i use matplotlib to plot and simulate links, joints and objectives, as well as ground and projection (terminal shadow!?).
My environment consist in one manipulator with 4 joints (3+terminal) with the following Denavit-Hartember parameters:

<p align="center"> 
<img src="https://i.imgur.com/IyulesQ.png"/>
</p>

<h3>The Objectives</h3>

The environment objectives consist in 10 random points refreshing when all points is colected.
Each point have one fixed reward and one variable reward, but that will be explained soon.

<p align="center"> 
<img src="https://media.giphy.com/media/Wonv0YvrM5Djy6XkXW/giphy.webp"/>
</p>

<h3>The Train</h3>

To create a new training file is necessary use the following command:

For PPO algorithm use -> `python3 ppo.py --train`

For DDPG algorithm use -> `python3 ddpg.py --train`

The training consist in a navigation among a large size of iterations with random values (angles). This is useful for gaining knowledge of the environment and understanding how work the observations. All of these events happen because the rewards.

<p align="center"> 
<img src="https://media.giphy.com/media/RM0A1YB58BWYvhH5fe/giphy.webp"/>
</p>

<h3>The Rewards</h3>
... ... ...

<p align="center"> 
<img src=""/>
</p>

<h2>The Algorithms</h2>

<h3>PPO</h3>
... ... ...

<p align="center"> 
<img src=""/>
</p>

<h3>DDPG</h3>
... ... ...

<p align="center"> 
<img src=""/>
</p>
