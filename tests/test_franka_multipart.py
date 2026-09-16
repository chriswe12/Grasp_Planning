import numpy as np
import pytest
from grasp_planning.rl.franka_multipart import part_assignment


def test_every_sampled_target_matches_spawned_geometry_and_global_coverage():
    target_parts=np.repeat(np.arange(46),np.arange(1,47))
    covered=set()
    for rank in range(4):
        parts,table,counts=part_assignment(target_parts,num_envs=32,rank=rank,world_size=4)
        covered.update(parts.tolist())
        for part,indices,count in zip(parts,table,counts):
            assert np.all(target_parts[indices[:count]]==part)
            assert len(set(indices[:count]))==count
    assert covered==set(range(46))


def test_heldout_geometry_only_is_assigned_when_other_parts_absent():
    targets=np.array([2,2,5,9,9,9])
    parts,table,counts=part_assignment(targets,num_envs=4,sequential=True)
    assert parts.tolist()==[2,5,9,2]
    for part,row,count in zip(parts,table,counts):
        assert set(row[:count])==set(np.flatnonzero(targets==part))


def test_insufficient_slots_fail_instead_of_silently_omitting_parts():
    with pytest.raises(ValueError,match='total environment slots'):
        part_assignment(np.arange(46),num_envs=8,world_size=4)
