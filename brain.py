STATE_SIZE = 10  # 7 joint positions + 3 target coordinates
ACTION_SIZE = 7  # 7 target joint positions

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),nn.ReLU(),
            nn.Linear(128, 256),nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )
    def forward(self, state):
        return self.net(state)
