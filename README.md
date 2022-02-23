# RL Snake Bot

<i>This project was adapted from Vedant Goswami's [tutorial](https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/) and [code](https://github.com/vedantgoswami/SnakeGameAI). Also Gobind Puniani adapted this into a tutorial cause he's a freakin G. </i>

This project showcases the power of reinforcement learning as an AI tool. Here, we use reinforcement learning to train a bot in a game of "Snake". In this arcade game, a player controls the direction of a blocky snake that doesn't stop moving. It grows in length as it eats food, which appears at a random spot on the screen. The object of the game is to keep the snake alive as along as possible by not letting it collide with a wall or intersect with itself. 

## Setup

Clone this repo onto your local machine. Create and activate a virtual environment, and then install the necessary packages (`pygame`, `torch`, etc.):
 - `python3 -m venv venv`
 - `source venv/bin/activate`
 - `pip install pygame`
 - `pip install torch`
 - `pip install` etc., as needed



The algorithm is divided into three Modules: <b>Agent</b> (`agent.py`), <b>Game</b> (`snake_gameai.py`), and <b>Model</b> (`model.py`).
  <p align='center'>
    <img src="https://github.com/vedantgoswami/SnakeGameAI/blob/main/Images/agentstate.PNG" width=400px height=290px>
  </p>
  <p>
    <img src="https://github.com/vedantgoswami/SnakeGameAI/blob/main/Images/game.png" width=390px height=250px align='left'>
    <img src="https://github.com/vedantgoswami/SnakeGameAI/blob/main/Images/model.png" width=390px height=250px align='right'>
  </p>

<br><br><br><br><br><br><br><br><br><br><br><br>
<hr />
<p>
  <h2>Result</h2>
<img src="https://github.com/vedantgoswami/SnakeGameAI/blob/main/Images/new.gif" width=380px height=250px align='left'>
<img src="https://github.com/vedantgoswami/SnakeGameAI/blob/main/Images/Animation.gif" width=380px height=250px align='right'>
<br><br><br><br><br><br><br><br><br><br><br>
<p style="font-size:25px">
<pre>              <b> Initial Epochs</b>                                           <b>After 100+ Epochs</b></pre>
</p>
