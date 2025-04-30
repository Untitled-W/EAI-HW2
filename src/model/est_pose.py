from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config


class EstPoseNet(nn.Module):
    
    """
    EstPoseNet is a neural network module designed to estimate the translation vector and rotation matrix directly from a given point cloud in the camera frame. It provides methods for forward propagation and estimation of pose parameters.

    Attributes
    config : Config
        Configuration object containing hyperparameters and settings for the network.

    Methods
    __init__(config: Config)
        Initializes the EstPoseNet with the given configuration.

    forward(pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs) -> Tuple[float, Dict[str, float]]
        Computes the forward pass of the network, calculating the loss and additional 
        metrics based on the ground truth translation and rotation.

    est(pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Estimates the translation vector and rotation matrix from the input point cloud.
    """

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        
        # 3--64--128--256
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        # 256--128--64--12
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )
        
        self.trans_loss = nn.MSELoss()
        self.rot_loss = nn.MSELoss()

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        
        # Encode the point cloud and get translation and rotation
        x = self.mlp1(pc)
        x = torch.max(x, dim=-2)[0]
        x = self.mlp2(x)
        pred_trans, pred_rot = x[:, :3], x[:, 3:].view(-1, 3, 3)
        
        # Use SVD to get the rotation matrix
        if torch.isnan(pred_rot).any():
            raise ValueError("Input contains NaN values")
        U, _, Vt = torch.linalg.svd(pred_rot)
        pred_rot = torch.matmul(U, Vt)
        # Ensure the determinant of the rotation matrix is 1
        det = torch.det(pred_rot)
        pred_rot[det < 0] *= -1
        
        # Compute the loss
        loss = self.config.trans_loss * self.trans_loss(pred_trans, trans) + \
                self.config.rot_loss * self.rot_loss(pred_rot, rot)
             
        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        # Encode the point cloud and get translation and rotation
        x = self.mlp1(pc)
        x = torch.max(x, dim=-2)[0]
        x = self.mlp2(x)
        pred_trans, pred_rot = x[:, :3], x[:, 3:].view(-1, 3, 3)
        
        # Use SVD to get the rotation matrix
        U, _, Vt = torch.linalg.svd(pred_rot)
        pred_rot = torch.matmul(U, Vt)
        # Ensure the determinant of the rotation matrix is 1
        det = torch.det(pred_rot)
        pred_rot[det < 0] *= -1
        
        return pred_trans, pred_rot