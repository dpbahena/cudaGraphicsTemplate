#include "CUDAHandler.h"
#include "simulationUI.h"
#include "GLManager.h"
#include "GL/freeglut.h"

#include <SDL2/SDL.h>
#include <chrono>
#include <thread>

GLManager* glManager;
CUDAHandler* cudaHandler;
SimulationUI simUI;


const int width =  1920;
const int height = 1080;

const int TARGET_FPS = 60;
const int FRAME_TIME_MS = 1000 / TARGET_FPS; // 16.67 ms per frame

struct MonitorInfo {
    int x, y;
    int width, height;
    std::string name;
};

std::vector<MonitorInfo> monitors;
int currentMonitor = 0;
bool moveWindow = false;

void findMonitors();
void imGuiMonitorSelector();

void idle() {
    glutPostRedisplay();
}

void display() {
    static auto lastFrameTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(now - lastFrameTime).count();
    lastFrameTime = now;

    // --- 1. Begin ImGui frame ---
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::NewFrame();
    
    imGuiMonitorSelector();

    simUI.render(*cudaHandler);

    
    // --- 2. Update simulation ---
    if (cudaHandler) {
        cudaHandler->updateDraw(deltaTime);
    }
    

    // --- 3. Render simulation scene ---
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (glManager) {
        glManager->render();
    }

    // --- 5. Render ImGui ---
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ImGui::Render();
    if (moveWindow && currentMonitor < monitors.size()) {
        MonitorInfo& m = monitors[currentMonitor];
        int posX = m.x + (m.width - width) / 2;  // Centered horizontally
        int posY = m.y + (m.height - height) / 2; // Centered vertically
        glutPositionWindow(posX, posY);
        moveWindow = false;
    }
    

    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    // --- 6. FPS control ---
    auto endTime = std::chrono::high_resolution_clock::now();
    int elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - now).count() / 1000;
    int sleepTime = FRAME_TIME_MS - elapsedTime;
    if (sleepTime > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime * 1000));
    }

    glutSwapBuffers();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    findMonitors();
    glutCreateWindow("CUDA OpenGL Sim");
    simUI.instance = &simUI;
    glManager = new GLManager(width, height);
    glManager->initializeGL();

    cudaHandler = new CUDAHandler(width, height, glManager->getTextureID());
    

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
   

    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL2_Init();
    ImGui::StyleColorsDark();

    // Input hooks
    glutKeyboardFunc(SimulationUI::keyboardCallback);
    // glutKeyboardUpFunc(ImGui_ImplGLUT_KeyboardUpFunc);
    // glutSpecialFunc(ImGui_ImplGLUT_SpecialFunc);
    // glutSpecialUpFunc(ImGui_ImplGLUT_SpecialUpFunc);
    
    glutMouseFunc(SimulationUI::mouseCallback);
    glutMouseWheelFunc(SimulationUI::mouseWheelCallback);
    glutMotionFunc([](int x, int y) { simUI.mouseMotionCallback(x, y); });
   
    

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    glutMainLoop();

    // Cleanup
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGLUT_Shutdown();
    ImGui::DestroyContext();

    return 0;
}

void findMonitors() {
    SDL_Init(SDL_INIT_VIDEO);
    int numDisplays = SDL_GetNumVideoDisplays();
    for (int i = 0; i < numDisplays; ++i) {
        SDL_Rect bounds;
        if (SDL_GetDisplayBounds(i, &bounds) == 0) {
            MonitorInfo m;
            m.x = bounds.x;
            m.y = bounds.y;
            m.width = bounds.w;
            m.height = bounds.h;
            m.name = "Monitor " + std::to_string(i + 1);
            monitors.push_back(m);
        }
    }

    if (monitors.size() >= 2) {
        // If 2 or more monitors, start on the second monitor
        MonitorInfo& m = monitors[1];
        glutInitWindowPosition(m.x + (m.width - width) / 2, m.y + (m.height - height) / 2);
        currentMonitor = 1; // Also set currentMonitor so ImGui knows
    } else if (!monitors.empty()) {
        // Only one monitor, start on it
        MonitorInfo& m = monitors[0];
        glutInitWindowPosition(m.x + (m.width - width) / 2, m.y + (m.height - height) / 2);
        currentMonitor = 0;
    }
    
    SDL_Quit();        
}


void imGuiMonitorSelector() {
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once); // Collapsed the first time
    ImGui::Begin("Monitor Selector");

    static int selectedMonitor = currentMonitor;
    if (ImGui::Combo("Choose Monitor", &selectedMonitor, [](void* data, int idx, const char** out_text) {
        const std::vector<MonitorInfo>* monitors = static_cast<const std::vector<MonitorInfo>*>(data);
        if (out_text) *out_text = monitors->at(idx).name.c_str();
        return true;
    }, (void*)&monitors, monitors.size())) {
        currentMonitor = selectedMonitor;
        moveWindow = true;
    }

    ImGui::End();
}


