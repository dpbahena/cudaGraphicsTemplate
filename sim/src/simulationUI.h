#pragma once

class CUDAHandler;  // Just temporary declaration but necessary

// float kernelMatrix[9] = {
//     0.5f, 1.0f, 0.5f,
//     1.0f, 0.0f, 1.0f,
//     0.5f, 1.0f, 0.5f
// };


class SimulationUI {
    public:

        void render(CUDAHandler &sim); // * real function
        void handleKeypress(unsigned char key, int x, int y); //* real function
        void handleMouseClick(int button, int state, int x, int y);
        static void keyboardCallback(unsigned char key, int x, int y); //* static wrapper
        static void mouseCallback(int button, int state, int x, int y); //* wrapper

        static void mouseMotionCallback(int x, int y);
        static void mouseWheelCallback(int wheel, int direction, int x, int y);

        static SimulationUI* instance;  //* Singleton-like access
        
        bool showMenu = true;


};