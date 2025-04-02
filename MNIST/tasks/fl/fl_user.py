from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader


@dataclass
class FLUser:
    user_id: int = 0
    train_loader: DataLoader = None
    test_loader: DataLoader = None
