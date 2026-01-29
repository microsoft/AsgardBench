import Magmathor.utils as Utils

MAX_JUMP_POSES = 25


class JumpCandidates:
    def __init__(self, obj):
        self.obj = obj
        self.interactable_poses = None
        self.center_angle = []
        self.reachable_poses = []
        self.poses_by_distance = []

    def calculate_center_angle(self, poses: list[dict]) -> float:
        # Extract angles from poses
        angles = [pose["rotation"] for pose in poses]
        center_angle = Utils.average_angles(angles)
        return center_angle

    def initialize(self):
        if self.interactable_poses is not None and len(self.interactable_poses) > 0:
            self.interactable_poses = self.order_poses(self.interactable_poses)
            if len(self.interactable_poses) > MAX_JUMP_POSES:
                self.interactable_poses = self.interactable_poses[:MAX_JUMP_POSES]

        if self.reachable_poses is not None and len(self.reachable_poses) > 0:
            self.reachable_poses = self.order_poses(self.reachable_poses)
            if len(self.reachable_poses) > MAX_JUMP_POSES:
                self.reachable_poses = self.reachable_poses[:MAX_JUMP_POSES]

    def order_poses(self, poses: list[dict]):
        if len(poses) == 0:
            return []

        self.center_angle = self.calculate_center_angle(poses)
        max_extra_index = 4
        extra_range = 0
        while extra_range < max_extra_index or len(self.poses_by_distance) == 0:
            filtered_poses = self.filter_poses_by_distance(poses, extra_range)
            extra_range += 0.5
            filtered_poses = Utils.sort_by_angle(filtered_poses, self.center_angle)
            self.poses_by_distance.append(filtered_poses)

            # Now remove the filterd poses from the original list
            poses = [pose for pose in poses if pose not in filtered_poses]

        # Now reassemble the poses back together
        sorted_poses = []
        for poses in self.poses_by_distance:
            sorted_poses.extend(poses)

        return sorted_poses

    def filter_poses_by_distance(self, poses, extra_range: float):

        obj_type = self.obj["objectType"]
        obj_pos = self.obj["position"]
        object_nppos = Utils.array_to_np(obj_pos)

        min_dist = Utils.min_interaction_distance(obj_type) - extra_range
        max_dist = Utils.max_interaction_distance(obj_type) + extra_range
        filtered_poses = []
        for pose in poses:
            pos = pose["position"]
            pos_nppos = Utils.array_to_np(pos)
            horizontal_distance = Utils.horizontal_distance(object_nppos, pos_nppos)
            if horizontal_distance < max_dist and horizontal_distance > min_dist:
                pose["extra_range"] = extra_range
                filtered_poses.append(pose)

        return filtered_poses
