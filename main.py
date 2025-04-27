from pyglet.gl import  *
import pyglet
from WarpRender import WarpRender
import time


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400,400)
        glClearColor(0.2,0.3,0.2,1.0)
        self.Render = WarpRender()
        self.Render.render()
        
    def on_draw(self):
        self.clear()
        glDrawElements(GL_TRIANGLES, len(self.Render.indices), GL_UNSIGNED_INT, None)
    
    def update(self,dt):
        self.Render.compute_and_render()


 
if __name__ == "__main__":
    
    total_count = 10000
    refresh_count = 0

    window = MyWindow(720, 720, "Shaotongf", resizable=True)

    def update_with_timer(dt):
        global refresh_count

        window.update(dt)
        refresh_count += 1

        if refresh_count >= total_count:
            total_time = time.time() - start_time
            print(f"total time is: {total_time:.6f} s")
            pyglet.app.exit()
            
    start_time = time.time()
    pyglet.clock.schedule_interval(update_with_timer, 1.0 / 200.0)  # 60/200 FPS
    pyglet.app.run()