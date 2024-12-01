import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

class ActionProjectorMixin:
    def __init__(self):
        """Initialize the ActionProjector."""
        pass

    @staticmethod
    def project_points(points, extrinsics, intrinsics):
        """
        Project 3D points into 2D using extrinsics and intrinsics.

        Args:
            points (np.ndarray): Array of shape (N, 3) or (3,) for a single point.
            extrinsics (np.ndarray): 4x4 extrinsics matrix.
            intrinsics (np.ndarray): 3x3 camera intrinsics matrix.

        Returns:
            np.ndarray: Projected 2D points, shape (N, 2) or (2,).
        """
        rvec = cv2.Rodrigues(np.array(extrinsics[:3, :3]))[0]
        tvec = np.array(extrinsics[:3, 3])
        projected, _ = cv2.projectPoints(np.array(points), rvec, tvec, np.array(intrinsics), np.zeros(5))
        return projected.squeeze()

    @staticmethod
    def draw_trajectory(image, trajectory, color=(0, 255, 0), thickness=2):
        """
        Draw the trajectory as a line on the image.

        Args:
            image (np.ndarray): The input image.
            trajectory (np.ndarray): Array of shape (N, 2) or (2,) for a single point.
            color (tuple): Color of the trajectory line.
            thickness (int): Thickness of the trajectory line.
        """
        for point in trajectory:
            pt = tuple(map(int, point))
            cv2.circle(image, pt, radius=3, color=color, thickness=thickness)

    @staticmethod
    def draw_orientation(image, points, orientations, extrinsics, intrinsics, scale=0.07, color=(255, 0, 0)):
        """
        Draw orientation arrows at specific points.

        Args:
            image (np.ndarray): The input image.
            points (np.ndarray): Array of shape (N, 3) or (3,) for a single point.
            orientations (np.ndarray): Array of shape (N, 3) or (3,) for a single orientation.
            extrinsics (np.ndarray): 4x4 extrinsics matrix.
            intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
            scale (float): Scale factor for arrow length.
            color (tuple): Color of the arrows.
        """
        # Ensure orientations have the correct shape (use only the first 3 components if quaternions are provided)
        orientations = orientations[:, :3]  # Extract the vector part if orientations are quaternions

        projected_points = ActionProjectorMixin.project_points(points, extrinsics, intrinsics)
        for i, (point, orientation) in enumerate(zip(projected_points, orientations)):
            # Calculate the end point for the arrow
            end_point_3d = points[i] + scale * orientation
            end_point_2d = ActionProjectorMixin.project_points(end_point_3d[np.newaxis, :], extrinsics, intrinsics)
            pt1 = tuple(map(int, point))
            pt2 = tuple(map(int, end_point_2d))
            cv2.arrowedLine(image, pt1, pt2, color, 2, tipLength=0.2)
            
    def visualize_actions(self, image, extrinsics, intrinsics, gt_positions, gt_orientations, pred_positions, pred_orientations, chunk_size=5):
        """
        Visualize the ground truth (GT) and predicted trajectories and orientations on the image.

        Args:
            image (np.ndarray): The input image.
            projection_matrix (np.ndarray): 3x4 projection matrix.
            gt_positions (np.ndarray): Array of shape (N, 3) representing GT 3D positions.
            gt_orientations (np.ndarray): Array of shape (N, 3) representing GT 3D orientations.
            pred_positions (np.ndarray): Array of shape (N, 3) representing predicted 3D positions.
            pred_orientations (np.ndarray): Array of shape (N, 3) representing predicted 3D orientations.
            chunk_size (int): Length of action chunks for visualization.
        """
        # Project and draw ground truth trajectory and orientations (green)
        gt_projected = self.project_points(gt_positions, extrinsics, intrinsics)
        self.draw_trajectory(image, gt_projected, color=(0, 255, 0))  # Green
        for i in range(0, len(gt_positions), chunk_size):
            chunk_positions = gt_positions[i:i + chunk_size]
            chunk_orientations = gt_orientations[i:i + chunk_size]
            self.draw_orientation(image, chunk_positions, chunk_orientations, extrinsics, intrinsics, color=(0, 255, 0))  # Green

        # Project and draw predicted trajectory and orientations (red)
        pred_projected = self.project_points(pred_positions, extrinsics, intrinsics)
        self.draw_trajectory(image, pred_projected, color=(255, 0, 0))  # Red
        for i in range(0, len(pred_positions), chunk_size):
            chunk_positions = pred_positions[i:i + chunk_size]
            chunk_orientations = pred_orientations[i:i + chunk_size]
            self.draw_orientation(image, chunk_positions, chunk_orientations, extrinsics, intrinsics, color=(255, 0, 0))  # Red

        # Show the final image with projections
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def visualize(self, batch, predicted_actions, epoch):
        # everything to cpu
        predicted_actions = {k: v.detach().cpu() for k, v in predicted_actions.items()}
        batch = {k: v.detach().cpu() for k, v in batch.items()}
        
        # Extract relevant data from ground truth and predicted dictionaries
        actions_franka_gt = batch['actions_franka']  # [BS, chunk_length, 7]
        actions_franka_pred = predicted_actions['actions_franka']  # [BS, chunk_length, 7]
        images_front = batch['oakd_front_view_images']  # [BS, 1, 3, 540, 960]
        images_side = batch['oakd_side_view_images']  # [BS, 1, 3, 540, 960]
        
        extrinsics_front = batch['oakd_front_view/extrinsics']  # [BS, 16]
        extrinsics_side = batch['oakd_side_view/extrinsics']  # [BS, 16]
        
        intrinsics_front = batch['oakd_front_view/intrinsics']  # [BS, 9]
        intrinsics_side = batch['oakd_side_view/intrinsics']  # [BS, 9]
        
        # Randomly sample an image and corresponding actions/projections
        idx = random.randint(0, actions_franka_gt.shape[0] - 1)
        idx = 45
        
        extrinsics_front = extrinsics_front[idx].reshape(4, 4)
        extrinsics_side = extrinsics_side[idx].reshape(4, 4)        
        
        intrinsics_front = intrinsics_front[idx].reshape(3, 3)
        intrinsics_side = intrinsics_side[idx].reshape(3, 3)
                
        extrinsics_front = np.linalg.inv(extrinsics_front)  
        
        extrinsics_side = np.linalg.inv(extrinsics_side)

        # Extract ground truth and predicted actions
        actions_gt = actions_franka_gt[idx]  # [chunk_length, 7]
        actions_pred = actions_franka_pred[idx]  # [chunk_length, 7]

        # Split into positions and orientations
        positions_gt = actions_gt[:, :3]  # [chunk_length, 3]
        orientations_gt = actions_gt[:, 3:]  # [chunk_length, 4]

        positions_pred = actions_pred[:, :3]  # [chunk_length, 3]
        orientations_pred = actions_pred[:, 3:]  # [chunk_length, 4]

        # Extract corresponding images
        image_front = images_front[idx, 0].permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC
        image_side = images_side[idx, 0].permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC

        # Normalize images to 0-255 for visualization
        image_front = (image_front * 255).astype(np.uint8)
        image_side = (image_side * 255).astype(np.uint8)
        
    
        frame_front = image_front.copy()
        frame_side = image_side.copy()
        
        # Visualize up to the current step in the trajectory
        self.visualize_actions(
            frame_front, extrinsics_front, intrinsics_front,
            positions_gt, orientations_gt,
            positions_pred, orientations_pred
        )
        self.visualize_actions(
            frame_side, extrinsics_side, intrinsics_side,
            positions_gt, orientations_gt,
            positions_pred, orientations_pred
        )
        
        cv2.imwrite("front.png", cv2.cvtColor(frame_front, cv2.COLOR_RGB2BGR))
        cv2.imwrite("side.png", cv2.cvtColor(frame_side, cv2.COLOR_RGB2BGR))

        return {"front_view" : "front.png", 
                "side_view" : "side.png"}

        