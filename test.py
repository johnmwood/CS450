class Ship:
    __init__(self, color, size, shape, velocity):
        self.color = color 
        self.size = size 
        self.shape = shape

        self.velocity = velocity 

    increase_speed(self): 
        self.velocity += 1


class Velocity: 
    __init__(self): 
        self.speed = 1 


s = Ship("green", "big", "triangle") 
print(s.shape)
