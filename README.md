
<h1 align="center">ManyTor</h1>
<h3 align="center">This is a project inspired by the gym module, we seeks to bring facilities to researchers and developers who work with robotic manipulators and/or reinforcement learning.</h3>

<p align="center"> 
  <img src="https://img.shields.io/badge/Vispy-v0.6.4-blue"/>
  <img src="https://img.shields.io/badge/Numpy-v1.18.2-blue"/>
  <img src="https://img.shields.io/badge/Tqdm-v4.42.1-blue"/>
</p>
<br/>

## Environment
<p align="justify"> 
  <img src="https://i.imgur.com/IyulesQ.png" alt="ManyTor" align="right" width="320">
  <a>For general purpose, i decided to create a new environmnet instead use one environment already created.
Is this project i use vispy to plot and simulate links, joints and objectives, as well as ground.
My base environment consist in one manipulator with 4 joints (3+terminal) with the Denavit-Hartember parameters present here on the right. </a>  
</p>
  
>**Obs**: I am currently working to generalize the code with the purpose of adapting the simulation to any manipulator model only by changing its forward kinematic function.

## Setup
<p align="justify"> 
 <a>All of requirements is show in the badgets above, but if you want to install all of them, enter the repository and execute the following line of code:</a>
</p>

```shell
pip3 install -r requirements.txt
```
## Objectives
<p align="justify"> 
  
  <a> The environment objectives consist in ![equation](https://latex.codecogs.com/gif.latex?x) random points refreshing when all points is colected. Each point have one fixed reward and one variable reward, but that will be explained soon.
  
</a>
</p>


## Reward System
<p align="justify" float="left"> 
  <img src="https://media.giphy.com/media/Izd6ZTUl6JvnjqH1a1/giphy.webp" alt="Neon Drive" align="right" width="320">
  

  As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state, and also returns a reward that indicates the consequences of  the action. In this task, rewards are +1 for every time the agent catches a ball and the environment terminates if all the balls were picked up or the manipulator hit the ground. This means that the most performing scenarios will end in ![equation](https://latex.codecogs.com/gif.latex?x) timesteps, where ![equation](https://latex.codecogs.com/gif.latex?x) means the number of objectives.
  
</p>

<p align="justify"> 
  <a><em>If you liked this repository, please don't forget to starred it!</em></a>  <img src="https://img.shields.io/github/stars/victorkich/Neon-Drive-Reinforcement-Learning?style=social" align="center"/>
</p>
