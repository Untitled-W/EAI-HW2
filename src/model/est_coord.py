from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        
        # self.linear1 = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.ReLU(),
        # )
        self.linear1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        # 3--64--128--256
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        # )
        self.mlp1 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(256+64, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3),
        # )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(256+64, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1, bias=False)
        )
        
        self.loss = nn.MSELoss()
        
    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        
        # pc: (B, N, 3) -> (B, 3, N)
        pc = pc.permute(0, 2, 1)
        x_1 = self.linear1(pc)
        x_2 = self.mlp1(x_1)
        x_3 = torch.max(x_2, dim=-1, keepdim=True)[0]
        x_3_expanded = x_3.expand(-1, -1, x_1.size(-1))
        x_4_input = torch.cat((x_1, x_3_expanded), dim=-2)
        x_4 = self.mlp2(x_4_input)
        pred_coord = x_4.permute(0, 2, 1)  # (B, N, 3)
        
        loss = self.loss(pred_coord, coord)

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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """

        # pc: (B, N, 3) -> (B, 3, N)
        pc = pc.permute(0, 2, 1)
        x_1 = self.linear1(pc)
        x_2 = self.mlp1(x_1)
        x_3 = torch.max(x_2, dim=-1, keepdim=True)[0]
        x_3_expanded = x_3.expand(-1, -1, x_1.size(-1))
        x_4_input = torch.cat((x_1, x_3_expanded), dim=-2)
        x_4 = self.mlp2(x_4_input)
        pred_coord = x_4.permute(0, 2, 1)  # (B, N, 3)
        pc = pc.permute(0, 2, 1)
        
        # Compute the centroid of the predicted coordinates and the input point cloud
        pred_centroid = pred_coord.mean(dim=1, keepdim=True)
        pc_centroid = pc.mean(dim=1, keepdim=True)

        # Center the predicted coordinates and the input point cloud
        pred_centered = pred_coord - pred_centroid
        pc_centered = pc - pc_centroid

        # Compute the covariance matrix
        covariance_matrix = torch.bmm(pc_centered.transpose(1, 2), pred_centered)

        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = torch.linalg.svd(covariance_matrix, full_matrices=False)
        UVt = U.matmul(Vt)
        det_UVt = torch.linalg.det(UVt)
        D = torch.diag_embed(torch.stack([
            torch.ones_like(det_UVt),
            torch.ones_like(det_UVt),
            det_UVt
        ], dim=-1))
        R = U.matmul(D).matmul(Vt)

        # Compute the translation vector
        t = pc_centroid - torch.bmm(R, pred_centroid.transpose(1,2)).transpose(1, 2)

        return t.squeeze(1), R  # (B, 3), (B, 3, 3)