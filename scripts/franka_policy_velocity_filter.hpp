#pragma once
#include <array>
#include <franka/lowpass_filter.h>
#include <franka/rate_limiting.h>

namespace franka_policy {
// The 100 Hz Servo output is held by JTC between updates. Smooth its edges at
// the 1 kHz hardware boundary, including zero targets on command expiry.
// This is a candidate tuning, not a claim of validated physical behavior.
constexpr double kVelocityCutoffHz = 10.0;
inline std::array<double, 7> filterVelocity(
    const std::array<double, 7>& target,
    const std::array<double, 7>& q_desired,
    const std::array<double, 7>& velocity_desired,
    const std::array<double, 7>& acceleration_desired) {
  std::array<double, 7> filtered{};
  for (size_t i = 0; i < filtered.size(); ++i) {
    filtered[i] = franka::lowpassFilter(franka::kDeltaT, target[i],
                                      velocity_desired[i], kVelocityCutoffHz);
  }
  return franka::limitRate(
      franka::computeUpperLimitsJointVelocity(q_desired),
      franka::computeLowerLimitsJointVelocity(q_desired),
      franka::kMaxJointAcceleration, franka::kMaxJointJerk,
      filtered, velocity_desired, acceleration_desired);
}
}  // namespace franka_policy
