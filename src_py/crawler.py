from src_py.utils import get_direction_offset, read_tile
from src_py.functions import evaluate


class Crawler:
    def __init__(self, pos_x, pos_y, world):
        self.x = pos_x
        self.y = pos_y
        self.world = world
        
        self.children = []
        self.isdead = False
        
        self.crawler_functions = {
            20: self.if_tile,
            21: self.die,
            22: self.goto,
        }
    

    def __call__(self):
        # if the crawler has children, remove the dead ones
        while self.children and self.children[0].isdead:
            self.children.pop()
        
        # if any children remain, have the first advance
        if self.children:
            self.children[0]()
            return

        self.execute()  
        self.step()       

       
    def step(self):
        direction, _ = read_tile(self.x, self.y, self.world)
        move_x, move_y = get_direction_offset(direction)
        # check step stays inside image
        self.x += move_x
        self.y += move_y


    def execute(self):
        _, instruction = read_tile(self.x, self.y, self.world)
        if instruction in self.crawler_functions:
            self.crawler_functions[instruction]()
            return
        
        evaluate(self.x, self.y, self.world)


    def if_tile(self):
        direction, _ = read_tile(self.x, self.y, self.world)
        cond_x, cond_y = get_direction_offset(direction)
        condition = evaluate(self.x + cond_x, self.y + cond_y, self.world)
        # check condition is bool
        if condition:
            move_x, move_y = get_direction_offset((direction+6) % 8)
        else:
            move_x, move_y = get_direction_offset((direction+2) % 8)

        self.x += move_x
        self.y += move_y


    def die(self):
        self.isdead = True

   
    def goto(self):
        direction, _ = read_tile(self.x, self.y, self.world)
        x_tile_offset = get_direction_offset((direction+6) % 8)
        y_tile_offset = get_direction_offset((direction+2) % 8)
        
        child_x = evaluate(self.x + x_tile_offset[0], self.y + x_tile_offset[1], self.world)
        child_y = evaluate(self.x + y_tile_offset[0], self.y + y_tile_offset[1], self.world)
        # check child_x, child_y are valid coordinates
        child = Crawler(child_x, child_y, self.world)
        self.children.append(child)