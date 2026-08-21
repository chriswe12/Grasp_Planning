"""RL-Games observers that keep distributed worker memory bounded."""

from __future__ import annotations

from rl_games.common.algo_observer import IsaacAlgoObserver


class DistributedSafeIsaacAlgoObserver(IsaacAlgoObserver):
    """Collect episode logging tensors only on the rank that writes them.

    RL-Games calls ``after_print_stats`` only on global rank zero. The stock
    Isaac observer nevertheless appends every rank's ``infos["episode"]``
    dictionary to ``ep_infos`` and relies on ``after_print_stats`` to clear it.
    On nonzero ranks that list therefore retains CUDA tensors indefinitely.
    """

    def after_init(self, algo):
        super().after_init(algo)
        self._collect_training_statistics = int(getattr(algo, "global_rank", 0)) == 0

    def process_infos(self, infos, done_indices):
        if not self._collect_training_statistics:
            return
        super().process_infos(infos, done_indices)

    def after_clear_stats(self):
        super().after_clear_stats()
        self.ep_infos.clear()
        self.direct_info.clear()


__all__ = ["DistributedSafeIsaacAlgoObserver"]
