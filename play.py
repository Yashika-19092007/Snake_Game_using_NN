import pygame
from neural_network import brain
from snake_game import snake_game
from vizualizer import network  # NAYA: Visualizer ko import kiya

# Pygame initialize karna zaroori hai display windows ke liye
pygame.init()

brain = brain.Brain()
game = snake_game.Main()

# NAYA: Visualizer ka object banaya aur usko AI ka dimaag (brain) de diya
visualizer = network.NetworkVisualizer(brain)

state = game.reset()
done = False
clock = pygame.time.Clock()

# Episode aur Score Variables (Visualizer ko dikhane ke liye)
episode = 1
reward_sum = 0

while not done: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
        # NAYA: Visualizer ki window close karne ka event handle karna
        visualizer.handle_event(event)

    # epsilon=0.0 taaki AI apna trained dimaag use kare
    action = brain.choose_action(state, epsilon=0.0)

    # Game ko ek step aage badhana
    state, reward, done = game.step(action)
    reward_sum += reward

    game.draw()
    
    # NAYA: Visualizer ko har step ke baad update karna
    visualizer.update(
        state=state, 
        action=action, 
        reward=reward_sum, 
        epsilon=0.0, 
        episode=episode, 
        score=game.score
    )

    clock.tick(10) # Speed of the game

pygame.quit()
print("Game Over! Final Score:", game.score)