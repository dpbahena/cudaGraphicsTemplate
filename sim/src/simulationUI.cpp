#include "simulationUI.h"
#include "CUDAHandler.h"


SimulationUI* SimulationUI::instance = nullptr;


void SimulationUI::render(CUDAHandler &sim)
{
    // Toggle with a TAB key to show menu
    if (ImGui::IsKeyPressed(ImGuiKey_Tab)) {
        showMenu = ! showMenu;
    }


    if (showMenu) {

        ImGui::Begin("Simulation Control");
        if (ImGui::Button("Reset Sim")) {
            sim.initGameLife();
        }
        if (sim.startSimulation)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.8f, 0.2f, 1.0f)); // green
        else
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f)); // red

        ImGui::Text("Start Simulation :"); ImGui::SameLine();
        if (ImGui::Button(sim.startSimulation ? "ON" : "OFF")) {
            sim.startSimulation ^= true;
        }
        ImGui::PopStyleColor();
        // Combo for Tree Display Mode
        const char* options[] = { "Random", "Vertical", "Horizontal", "Checkered", "Diagonal", "X-shape", "Circle", "Spiral"};
        static int selectedOption = sim.option; 

        if (ImGui::Combo("Game of Life Pattern", &selectedOption, options, IM_ARRAYSIZE(options))) {
            sim.option = selectedOption;
        }
        if (sim.option > 0)
            ImGui::SliderFloat("WidthFactor", &sim.widthFactor, 0.01f,0.4f);
        ImGui::Separator;
        ImGui::Text("Total Cells: %d", (int)sim.gamelife.size());
        ImGui::Text("Zoom Factor: %f", sim.zoom);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Actual FPS: %.1f", 1.0f / sim.dt);
        
        
        // Add more sliders for simulation params here
        ImGui::End();

    }
}  
    
    


    // if (showMenu) {

    //     ImGui::Begin("Simulation Control");
        
        // const char* modes[] = {"Editing", "Simulating"};
        // static int currentMode = 0;
        // if (ImGui::Combo("Mode", &currentMode, modes, IM_ARRAYSIZE(modes))) {
        //     sim.setMode(currentMode == 0 ? SimMode::Editing : SimMode::Simulating);
        // }
        // ImGui::Separator;

        // if (sim.isEditing()) {
        //     ImGui::Text("Editor Mode: add planets, set orbits.");
        //     // TODO: future add buttons to place/edit bodies
        // } else {
        //     ImGui::Text("Simulation MOde: gravity in action.");
        //     // TODO: add sliders or pause control
        // }
        // if (ImGui::Button("Reset Sim")) {
        //     sim.resetSimulation();
        // }
        // if (ImGui::IsItemHovered()) {
        //     ImGui::SetTooltip("Remove all particles & reset the simulation");
        // }
        // if (sim.getTreeMode() == NBODY) {
        //     ImGui::SliderInt("Trail Length", &sim.trailLength, 1, 400);
        // }else {
        //     ImGui::SliderInt("Trail Length", &sim.trailLength, 1, 60);
        // }

        // ImGui::SliderInt("Stiffness", &sim.stiffness, 1, 1500);
        // ImGui::SliderFloat("StringLength", &sim.restLength, 0.1, 200.0f);
        // ImGui::SliderFloat("Radius", &sim.particleRadius, 0.5f, 20.0f);
        // ImGui::SliderFloat("Mass", &sim.particleMass, 0.1f, 10.0f);
        // ImGui::SliderFloat("Damping", &sim.damping, 1.0f, 300.0f);
        // ImGui::SliderFloat("Kpressure", &sim.kp, 0.01f, 10.0f);
        // ImGui::SliderFloat("Correction", &sim.posCorrection, 0.01f, 0.5f);
        // ImGui::SliderInt("# of Correction Passes", &sim.posCorrectionPasses, 0, 10);
        // ImGui::SliderInt("Overlap correction passes", &sim.overlapCorrectionPasses, 0, 20);
        // if(sim.getTreeMode() == NBODY) {
        //     ImGui::SliderFloat("G Value", &sim.gValue, 1, 5000);
        //     ImGui::SliderFloat("Min Value", &sim.minValue, 0, 1000);
        //     ImGui::SliderFloat("Max VAlue", &sim.maxValue, 1, 5000);
        // }
        // if(sim.getTreeMode() == SPHERE) {
        //     ImGui::SliderInt("Circular Boundary", &sim.circularBoundary, 100, 500);
        //     ImGui::SliderFloat("Restitution", &sim.restitution, -0.5f, 2.0f);
        //     ImGui::SliderFloat("Friction", &sim.friction, -0.5f, 2.0f);
        // }
        // ImGui::PushID("Drag");
        // ImGui::SliderFloat("##Slider", &sim.dragCoeff, 0.000f, 2.0f, "Drag Coef: %.3f");
        // ImGui::SameLine();
        // ImGui::InputFloat("##Input", &sim.dragCoeff);
        // ImGui::PopID();
        // ImGui::SliderFloat("West wind", &sim.wind, -20.0f, 20.0f);
        // ImGui::SliderFloat("Valley breeze", &sim.valleyBreeze, 0.f, 40.0f);
        // ImGui::SliderInt("Cursor Radius", &sim.mouseCursorRadius, 5, 50);
        // ImGui::Checkbox("Gravity", sim.getGravityPtr());
        // ImGui::Checkbox("FollowMode", sim.getFollowModePtr());
        
        // int toolMode = static_cast<ToolMode>(sim.getToolMode());
        // ImGui::Text("Select tool:"); 
        // ImGui::RadioButton("Disturbe", &toolMode, DISTURBE); ImGui::SameLine();
        // ImGui::RadioButton("Tear", &toolMode, TEAR); ImGui::SameLine();
        // ImGui::RadioButton("Pull/drag", &toolMode, DRAG); ImGui::SameLine();
        // ImGui::RadioButton("Activate", &toolMode, ACTIVE);

        
        
        // sim.setToolMode(toolMode);
        
        // // Combo for Tree Display Mode
        // const char* treeModes[] = { "Cycle Tree", "Complete Tree", "In Line", "Grid", "Single", "Nbody", "Spider", "Pendulum", "Sphere"};
        // static int selectedTreeMode = sim.getTreeMode(); // 0: Cycle, 1: Complete, 2: In Line

        // if (ImGui::Combo("Tree Mode", &selectedTreeMode, treeModes, IM_ARRAYSIZE(treeModes))) {
        //     sim.setTreeMode(selectedTreeMode);
        // }
        // ImGui::Checkbox("Show Links", sim.getShowLinksPtr());
        // ImGui::Checkbox("Show Particles", sim.getShowParticlesPtr());
        // if (sim.getTreeMode() == GRID) {
        //     ImGui::Checkbox("Show Texture", sim.getTextureModPtr());
        // }
        // ImGui::Checkbox("Windy", &sim.windy);


        // ImGui::Checkbox("Show Trails", sim.getShowTrailsPtr());
        // ImGui::Text("# of Particles: %d",(int)sim.particles.size());
        // ImGui::Text("Zoom Factor: %f", sim.zoom);
    //     ImGui::Separator;
    //     ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    //     ImGui::Text("Actual FPS: %.1f", 1.0f / sim.dt);
        

    //     // Add more sliders for simulation params here
    //     ImGui::End();
    // }
// }

void SimulationUI::handleKeypress(unsigned char key, int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) return;

    if (key == 27){
        exit(0); 
    }
    // ? Handle other keys
}

void SimulationUI::keyboardCallback(unsigned char key, int x, int y)
{
    // Step 1: let ImGui handle the keyboard input
    ImGui_ImplGLUT_KeyboardFunc(key, x, y);

    // Step 2: Pass it to your own logic
    if (instance){
        instance->handleKeypress(key, x, y);
    }
}

void SimulationUI::mouseCallback(int button, int state, int x, int y)
{
    // Step 1: Let ImGui handle input (so UI stays interactive)
    ImGui_ImplGLUT_MouseFunc(button, state, x, y);

    // Step 2: Pass it to your own logic
    if (instance){
        instance->handleMouseClick(button, state, x, y);
    }


}

void SimulationUI::mouseMotionCallback(int x, int y)
{
    
    CUDAHandler* sim = CUDAHandler::instance;
    // * Let ImGui handles mouse events first
    ImGui_ImplGLUT_MotionFunc(x, y);

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse || !sim) return;

    int mods = glutGetModifiers();
    if (sim->isDragging  && sim->isPanEnabled) {
        float dx = (x - sim->lastMouseX) / sim->zoom;
        float dy = (y - sim->lastMouseY) / sim->zoom;

        sim->panX += dx;
        sim->panY += dy;

        sim->lastMouseX = x;
        sim->lastMouseY = y;

        glutPostRedisplay();
    } else if (sim->toolMode == DISTURBE) {
        // Convert screen coordinates to world space
        float worldx = (x - sim->width / 2.0f) / sim->zoom - sim->panX;
        float worldy = (y - sim->height / 2.0f) / sim->zoom - sim->panY;
        vec2 targetPos(worldx, worldy);
        sim->disturbeGameLife(targetPos);
        // record screen position
        // sim->lastMouseX = x;
        // sim->lastMouseY = y;

    }
    sim->lastMouseX = x;
    sim->lastMouseY = y;

}

void SimulationUI::mouseWheelCallback(int wheel, int direction, int x, int y)
{
    if (!CUDAHandler::instance) return;
    int mods = glutGetModifiers();  // activate zoom/via wheel mouse if ALT key is pressed
    if (mods & GLUT_ACTIVE_SHIFT) { 
        float zoomFactor = (direction > 0) ? 1.1f : 1.0f / 1.1f;
        CUDAHandler::instance->zoom *= zoomFactor;
        glutPostRedisplay();
    }
}

void SimulationUI::handleMouseClick(int button, int state, int x, int y){
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse || !CUDAHandler::instance) return;

    CUDAHandler* sim = CUDAHandler::instance;
    

    // ✅ Convert screen coordinates to world/sim space
    float worldX = (x - sim->width / 2.0f) / sim->zoom - sim->panX;
    float worldY = (y - sim->height / 2.0f) / sim->zoom - sim->panY;
    // vec2f mousePos(worldX, worldY);
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {

        int mods = glutGetModifiers();
        if (mods & GLUT_ACTIVE_SHIFT) { // onnly activate dragging if SHIFT key is pressed
            sim->isDragging = true;
            sim->isPanEnabled = true;
            sim->lastMouseX = x;
            sim->lastMouseY = y;
            

        } else {
            sim->toolMode = DISTURBE;
        }

        // if (sim->getToolMode() == DRAG)
        //     sim->applyPullToParticle(mousePos); // grab a particle
        
       
        // sim->lastMouseX = x; // This is used to update the circular mouse pointer 
        // sim->lastMouseY = y; 

        // sim->leftMouseDown = true;S

        
    } else if (state == GLUT_UP) {
        sim->isDragging = false;
        sim->isPanEnabled = false;
        // sim->leftMouseDown = false;
        // sim->selectedParticle = nullptr;
        // for (int i = 0; i < sim->selectedParticles.size(); ++i){
        //     sim->particles[sim->selectedParticles[i]].color = sim->simulationParticlesColor;
        // }
        // sim->selectedParticles.clear();
        // sim->setToolMode(DISTURBE); // default

    }

    // if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {

    //     // ✅ Convert screen coordinates to world/sim space
    //     // float worldX = (x - sim->width / 2.0f) / sim->zoom - sim->panX;
    //     // float worldY = (y - sim->height / 2.0f) / sim->zoom - sim->panY;

    //     // float mass = random_double(1.0f, 10.0);
    //     // float radius = random_double(5.0, 20.0);
    //     // float indexColor = random_int(0, sim->colorPalette.size());

    //     // Particle ball(Vec2(worldX, worldY), mass, radius , sim->colorPalette[indexColor], 0);
    //     // if (!sim->particles.empty()) {
    //     //     // ball.velocity = Vec2(0.0f, 0.0f);

    //     //     sim->particles.push_back(ball);
        
    //     //     // update trail size
    //     //     sim->particleTrail.resize(sim->particles.size());
    //     //     // sim->planetAngle.resize(sim->particles.size());
    //     // }
        
           

    // }
}