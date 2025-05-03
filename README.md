# Game of Life

## Ubuntu 24.01

### Needs:ImGui, GLUT, CUDA, and OpenGL 
* IMGUI folder is empty. Create a new one
* Delete folder
* from terminal go to /sim
* git clone https://github.com/ocornut/imgui.git


## Compile & Run
### Create the build directory (if not already there)
cd sim
mkdir -p build
cd build
### Run cmake to configure the build system
cmake ..
### Build the project
cmake --build .
#### Or, if you prefer make directly:
make
### Run your program
./your_executable_name


## Pan & Zoom
* Pan: SHIFT + Left Mouse
* Zoom: SHIFT + Wheel Mouse
## Dragging Objects
* Left Click: Disturbe cells in Game of Life







