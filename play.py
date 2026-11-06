import pygame
from neural_network import brain
from snake_game import snake_game
from vizualizer import network  


pygame.init()

brain = brain.Brain()
game = snake_game.Main()


visualizer = network.NetworkVisualizer(brain)

state = game.reset()
done = False
clock = pygame.time.Clock()


episode = 1
reward_sum = 0

while not done: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
        
        visualizer.handle_event(event)


    action = brain.choose_action(state, epsilon=0.0)


    state, reward, done = game.step(action)
    reward_sum += reward

    game.draw()
    
    
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