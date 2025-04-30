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
        
        # # 3--64--128--256
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),
        # )
        # # 256--128--64--12
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 12),
        # )
        
        self.shared_mlp = nn.Sequential(        # input: (B, 3, N)
            nn.Conv1d(3, 64, 1, bias=False),    # -> (B, 64, N)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1, bias=False),  # -> (B, 128, N)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 1, bias=False), # -> (B, 256, N)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)     # -> (B, 256)
        # for translation vector
        self.head_t = nn.Sequential(
            nn.Linear(256, 128, bias=False),    # -> (B, 128)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),                  # -> (B, 3)
        )
        # for rotation matrix
        self.head_r = nn.Sequential(
            nn.Linear(256, 128, bias=False),    # -> (B, 128)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9),                  # -> (B, 9)
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
        # x = self.mlp1(pc)
        # x = torch.max(x, dim=-2)[0]
        # x = self.mlp2(x)
        # pred_trans, pred_rot = x[:, :3], x[:, 3:].view(-1, 3, 3)
        
        x = pc.permute(0, 2, 1)         # -> (B, 3, N)
        f = self.shared_mlp(x)          # -> (B, 256, N)
        f = self.pool(f).squeeze(-1)    # -> (B, 256)

        pred_trans = self.head_t(f)         # -> (B, 3)
        pred_rot = self.head_r(f)

        # Use SVD to get the rotation matrix
        if torch.isnan(pred_rot).any():
            raise ValueError("Input contains NaN values")
        # U, _, Vt = torch.linalg.svd(pred_rot)
        # pred_rot = torch.matmul(U, Vt)
        # # Ensure the determinant of the rotation matrix is 1
        # det = torch.det(pred_rot)
        # pred_rot[det < 0] *= -1
        
        U, S, Vt = torch.linalg.svd(pred_rot.view(-1, 3, 3), full_matrices=False)
        UVt = U.matmul(Vt)
        det_UVt = torch.linalg.det(UVt)
        D = torch.diag_embed(torch.stack([
            torch.ones_like(det_UVt),
            torch.ones_like(det_UVt),
            det_UVt
        ], dim=-1))
        pred_rot = U.matmul(D).matmul(Vt)

        trans_loss = self.trans_loss(pred_trans, trans)
        rot_loss = self.rot_loss(pred_rot, rot)
        
        # Compute the loss
        loss = self.config.trans_loss * trans_loss + \
                self.config.rot_loss * rot_loss
             
        metric = dict(
            loss=loss,
            trans_loss=trans_loss,
            rot_loss=rot_loss,
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
        # x = self.mlp1(pc)
        # x = torch.max(x, dim=-2)[0]
        # x = self.mlp2(x)
        # pred_trans, pred_rot = x[:, :3], x[:, 3:].view(-1, 3, 3)
        
        x = pc.permute(0, 2, 1)         # -> (B, 3, N)
        f = self.shared_mlp(x)          # -> (B, 256, N)
        f = self.pool(f).squeeze(-1)    # -> (B, 256)

        pred_trans = self.head_t(f)         # -> (B, 3)
        pred_rot = self.head_r(f)

        # Use SVD to get the rotation matrix
        U, S, Vt = torch.linalg.svd(pred_rot.view(-1, 3, 3), full_matrices=False)
        UVt = U.matmul(Vt)
        det_UVt = torch.linalg.det(UVt)
        D = torch.diag_embed(torch.stack([
            torch.ones_like(det_UVt),
            torch.ones_like(det_UVt),
            det_UVt
        ], dim=-1))
        pred_rot = U.matmul(D).matmul(Vt)
        
        return pred_trans, pred_rot