"""Race-safe, single-worker policy-to-pickup request."""

import threading


class PickupHandoff:
    def __init__(self):
        self._lock = threading.Lock()
        self._phase = "loading"
        self._requested = False

    def policy_started(self):
        with self._lock:
            self._phase = "policy"

    def available(self):
        with self._lock:
            return self._phase == "policy" and not self._requested

    def request(self):
        with self._lock:
            if self._phase != "policy" or self._requested:
                return False
            self._requested = True
            return True

    def requested(self):
        with self._lock:
            return self._requested

    def finish_policy(self):
        with self._lock:
            self._phase = "finishing"
            return self._requested
