// Read-only protocol/model probe. Never starts a control loop or sends commands.
#include <franka/robot.h>
#include <franka/model.h>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    // Reading feedback does not require realtime scheduling. Motion still does.
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    const auto state = robot.readOnce();
    auto model = robot.loadModel();
    const auto gravity = model.gravity(state);
    std::cout << "Server protocol: " << robot.serverVersion() << "\nJoints:";
    for (auto q : state.q) std::cout << " " << q;
    std::cout << "\nModel gravity:";
    for (auto g : gravity) std::cout << " " << g;
    std::cout << "\nRead-only feedback and model OK; no control started.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
