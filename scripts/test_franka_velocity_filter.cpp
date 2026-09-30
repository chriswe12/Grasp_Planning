// Offline only: no Robot instance, network connection, or hardware commands.
#include "franka_policy_velocity_filter.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
int main() {
  std::array<double, 7> q{0, 0, 0, -1.5, 0, 1.5, 0}, v{}, a{}, target{};
  double first = 0, stop100 = 0, peak = 0;
  for (int n = 0; n < 1500; ++n) {
    target.fill(n < 1000 ? 0.05 : 0.0);
    auto next = franka_policy::filterVelocity(target, q, v, a);
    if (n == 0) first = next[0];
    if (n == 1100) stop100 = std::abs(next[0]);
    for (size_t i = 0; i < 7; ++i) {
      assert(std::isfinite(next[i]));
      assert(next[i] >= -1e-9 && next[i] <= 0.05 + 1e-9);
      double acc = (next[i] - v[i]) / .001;
      assert(std::abs(acc) <= franka::kMaxJointAcceleration[i] + 1e-6);
      assert(std::abs((acc - a[i]) / .001) <= franka::kMaxJointJerk[i] + 1e-4);
      a[i] = acc;
      q[i] += next[i] * .001;
      peak = std::max(peak, std::abs(next[i]));
    }
    v = next;
  }
  assert(first < .005 && peak > .049);
  assert(stop100 < .0002 && std::abs(v[0]) < 1e-9);
  // Reject corrupt input rather than passing it through to the driver.
  target[0] = std::numeric_limits<double>::quiet_NaN();
  bool rejected = false;
  try { franka_policy::filterVelocity(target, q, v, a); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  std::cout << "PASS: no step overshoot; acceleration/jerk limits; zero settles; NaN rejected\n"
            << "first_step_rad_s=" << first << " stop_after_100ms_rad_s=" << stop100 << '\n';
}
