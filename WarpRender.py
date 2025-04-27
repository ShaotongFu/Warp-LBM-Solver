import ctypes
import warp as wp
import numpy as np
from pyglet.gl import *
from PIL import Image 
import math

grid_width = wp.constant(1024)
grid_height = wp.constant(512)

inlet_velocity = wp.constant(wp.vec2(0.04, 0))
niu = wp.constant(0.01)

dimension = wp.constant(9)
weight = [4.0/9.0] + [1.0/9.0] * 4 + [1.0/36.0] * 4
c = [
    wp.vec2(0.0, 0.0),
    wp.vec2(1.0, 0.0),
    wp.vec2(0.0, 1.0),
    wp.vec2(-1.0, 0.0),
    wp.vec2(0.0, -1.0),
    wp.vec2(1.0, 1.0),
    wp.vec2(-1.0, 1.0),
    wp.vec2(-1.0, -1.0),
    wp.vec2(1.0, -1.0)
]

wp.config.quiet = True
vec9d = wp.types.vector(length=9, dtype=wp.float32)

if not wp.get_cuda_device_count():
    print(
        "Some snippets in this notebook assume the presence of "
        "a CUDA-compatible device and won't run correctly without one."
    )

@wp.func
def period(value: int, range: int):
    return (value + range) % range

@wp.kernel
def init_function(f0: wp.array3d(dtype=float), weight: wp.array(dtype=float)):
    i, j = wp.tid()

    for k in range(9):
        f0[k, i, j] =  weight[k]
        
@wp.kernel
def stream_and_collision(f0: wp.array3d(dtype=float),  f1: wp.array3d(dtype=float), c: wp.array(dtype=wp.vec2), 
                         u: wp.array2d(dtype=wp.vec2), mag_u: wp.array(dtype=float),p: wp.array2d(dtype=float), weight: wp.array(dtype=float), mark: wp.array2d(dtype=int)):
    i, j = wp.tid()

    feq = vec9d()
    F = vec9d()
    
    #streaming
    F[0] = f0[0, i, j]
    F[1] = f0[1, period(i - 1, grid_width), j]
    F[2] = f0[2, i, period(j - 1, grid_height)]
    F[3] = f0[3, period(i + 1, grid_width), j]
    F[4] = f0[4, i, period(j + 1, grid_height)]
    F[5] = f0[5, period(i - 1, grid_width), period(j - 1, grid_height)]
    F[6] = f0[6, period(i + 1, grid_width), period(j - 1, grid_height)]
    F[7] = f0[7, period(i + 1, grid_width), period(j + 1, grid_height)]
    F[8] = f0[8, period(i - 1, grid_width), period(j + 1, grid_height)]

    #boundary condition
    # up wall
    if(j == grid_height - 1):
        F[4] = f0[2, i, j]
        F[7] = f0[5, i, j]
        F[8] = f0[6, i, j]

    #bottom wall
    if(j == 0):
        F[2] = f0[4, i, j]
        F[5] = f0[7, i, j]
        F[6] = f0[8, i, j]

    # outlet
    if(i == grid_width - 1):
        for k in range(9):
            f1[k, i, j] =  f0[k, i - 1, j]

        return

    #inlet
    if(i == 0):
        for k in range(9):
            feq[k] =  f0[k, i + 1, j]

        rho = feq[0] + feq[1] + feq[2] + feq[3] + feq[4] + feq[5] + feq[6] + feq[7] + feq[8]
        new_u = wp.vec2((feq[1] +  feq[5] + feq[8] - feq[3] -  feq[6] - feq[7])/rho, (feq[2] +  feq[5] + feq[6] - feq[4] -  feq[7] - feq[8])/rho)
        u_squre = 1.5 * wp.dot(new_u, new_u)
        feq[1] = feq[1] - rho * weight[1] * (1.0 + 3.0 * wp.dot(c[1], new_u) + 4.5 *wp.dot(c[1], new_u)* wp.dot(c[1], new_u) - u_squre)
        feq[5] = feq[5] - rho * weight[5] * (1.0 + 3.0 * wp.dot(c[5], new_u) + 4.5 *wp.dot(c[5], new_u)* wp.dot(c[5], new_u) - u_squre)
        feq[8] = feq[8] - rho * weight[8] * (1.0 + 3.0 * wp.dot(c[8], new_u) + 4.5 *wp.dot(c[8], new_u)* wp.dot(c[8], new_u) - u_squre)
        
        new_u = inlet_velocity
        u_squre = 1.5 * wp.dot(new_u, new_u)
        for k in range(9):
            f1[k, i, j] = feq[k] + rho * weight[k] * (1.0 + 3.0 * wp.dot(c[k], new_u) + 4.5 *wp.dot(c[k], new_u)* wp.dot(c[k], new_u) - u_squre)

        return
        
    #configuration
    if( mark[grid_height - 1 - j , i] == 0):
        temp = F[1]
        F[1] = F[3]
        F[3] = temp

        temp = F[2]
        F[2] = F[4]
        F[4] = temp

        temp = F[5]
        F[5] = F[7]
        F[7] = temp

        temp = F[6]
        F[6] = F[8]
        F[8] = temp

        for k in range(9):
            f1[k, i, j] =  F[k]

        return 
    
    #collision
    rho = F[0] + F[1] + F[2] + F[3] + F[4] + F[5] + F[6] + F[7] + F[8]
    new_u = wp.vec2((F[1] +  F[5] + F[8] - F[3] -  F[6] - F[7])/rho, (F[2] +  F[5] + F[6] - F[4] -  F[7] - F[8])/rho)
    tau_inv = 1.0 / (3.0 * niu + 0.5)

    u[i, j] = new_u
    p[i, j] = rho / 3.0
    mag_u[j * grid_width + i] = wp.length(new_u)/length(1.5*inlet_velocity)

    u_squre = 1.5 * wp.dot(new_u, new_u)
    
    for k in range(9):
        feq[k] = rho * weight[k] * (1.0 + 3.0 * wp.dot(c[k], new_u) + 4.5 *wp.dot(c[k], new_u)* wp.dot(c[k], new_u) - u_squre)
        f1[k, i, j] =  F[k] * (1.0 - tau_inv) + feq[k] * tau_inv



class WarpSolver:
    def __init__(self):
        self.shape = (grid_width, grid_height)
        f_shape = (dimension, grid_width, grid_height)

        self.u = wp.zeros(self.shape, dtype=wp.vec2)
        self.p = wp.zeros(self.shape, dtype=float)
        self.mag_u = wp.zeros(grid_width*grid_height, dtype=float)

        self.f0 = wp.zeros(f_shape, dtype=float)
        self.f1 = wp.zeros(f_shape, dtype=float)

        self.weight = wp.array(weight,dtype=wp.float32)
        self.c = wp.array(c, dtype=wp.vec2)

        wp.launch(init_function, dim=self.shape, inputs=[self.f0, self.weight])
                       
        self.read_figure()


    def read_figure(self):
        image = Image.open('nvidia.bmp').convert('L')
        img_array = np.array(image)
        self.mark = wp.array2d(img_array, dtype=wp.int32)

    def lb_step(self):
        
        # with wp.ScopedTimer("step"):   
            for _ in range(100):
                wp.launch(stream_and_collision, dim=self.shape, inputs=[self.f0, self.f1, self.c, self.u, self.mag_u, self.p, self.weight, self.mark])
            
                (self.f0, self.f1) = (self.f1, self.f0)     

    def step_and_render(self, frame_num=None, img=None):
        self.lb_step()

        with wp.ScopedTimer("render"):
            if img:
                img.set_array(self.mag_u.numpy())

        return (img,)



class WarpRender:

    def __init__(self):    
        wp.init
        
        self.nx = grid_width
        self.ny = grid_height
        self.triangle_points, self.indices = self.create_vertex_grid_vectorized(self.nx, self.ny)
        self.solver = WarpSolver()

        self.vertex_shader_source = b"""
        #version 330
        in layout(location = 0) vec3 position;
        in layout(location = 1) float value;
        
        out float scalarValue;
        void main()
        {
            gl_Position = vec4(position, 1.0f);
            scalarValue = value;
        }
        """
        self.fragment_shader_source = b"""
        #version 330
        in float scalarValue;
        
        out vec4 outColor;

        vec3 heatmapColor(float value) {
            vec3 color;
            if (value < 0.25) {
                color = mix(vec3(0, 0, 1), vec3(0, 1, 1), value * 4.0);
            } else if (value < 0.5) {
                color = mix(vec3(0, 1, 1), vec3(0, 1, 0), (value - 0.25) * 4.0);
            } else if (value < 0.75) {
                color = mix(vec3(0, 1, 0), vec3(1, 1, 0), (value - 0.5) * 4.0);
            } else {
                color = mix(vec3(1, 1, 0), vec3(1, 0, 0), (value - 0.75) * 4.0);
            }
            return color;
        }        
        void main()
        {
            outColor = vec4(heatmapColor(scalarValue), 1.0f);
        }
        """
        
        vertex_buff = ctypes.create_string_buffer(self.vertex_shader_source)
        c_vertex = ctypes.cast(ctypes.pointer(ctypes.pointer(vertex_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, 1, c_vertex, None)
        glCompileShader(vertex_shader)
        
        fragment_buff = ctypes.create_string_buffer(self.fragment_shader_source)
        c_fragment = ctypes.cast(ctypes.pointer(ctypes.pointer(fragment_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, 1, c_fragment, None)
        glCompileShader(fragment_shader)

        shader = glCreateProgram()
        glAttachShader(shader,vertex_shader)
        glAttachShader(shader,fragment_shader)
        glLinkProgram(shader)
                    
        glUseProgram(shader)

    def render(self):
        ibo = GLuint(0)
        glGenBuffers(1, ibo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.indices)*4, (GLuint * len(self.indices))(*self.indices), GL_STATIC_DRAW)
 
        vbos = GLuint * 2  # 创建两个VBO
        self.vbo_ids = vbos()
        glGenBuffers(2, self.vbo_ids)
        
        # poistion
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_ids[0])
        glBufferData(GL_ARRAY_BUFFER, len(self.triangle_points)*4, (GLfloat * len(self.triangle_points))(*self.triangle_points), GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        #Color Bind
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_ids[1])
        self.gl_data = np.zeros(self.nx*self.ny, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, self.gl_data.nbytes, self.gl_data.ctypes.data, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1,1,GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)  
        
        self.cuda_gl_buffer = wp.RegisteredGLBuffer(self.vbo_ids[1])

    def create_vertex_grid_vectorized(self, width, height):
        vertices = []
        # x is normalized by [-1, 1]
        # y is normalized by [-ny/nx, ny/nx]
        for i in range(height):
            for j in range(width):
                vertices.extend([float(j)/(width-1)*2.0 - 1.0, float(i)/(height-1)*(height/width)*2.0 - (height/width), 0.0])   
        indices = []
        for i in range(height-1):
            for j in range(width-1):
                v0 = i * width + j            
                v1 = i * width + (j + 1)      
                v2 = (i + 1) * width + (j + 1)  
                v3 = (i + 1) * width + j      
            
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        return vertices, indices

    # change the kernel for visualization 
    def compute_and_render(self):
        self.solver.mag_u = self.cuda_gl_buffer.map(dtype=wp.float32, shape=(self.nx*self.ny,))
        self.solver.lb_step()
        self.cuda_gl_buffer.unmap() 
        glBindBuffer(GL_ARRAY_BUFFER, 0)



