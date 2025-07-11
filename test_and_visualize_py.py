import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import numpy as np
import imageio
from IPython.display import Video

# --- Define the PolicyNetwork ---
STATE_SIZE = 10  # Joint states + target position (7 + 3)
ACTION_SIZE = 7  # Number of joints in the Kuka arm

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

# --- Load the Trained Model ---
try:
    brain_file = "/content/kuka_brain.pth"
    brain = PolicyNetwork()
    brain.load_state_dict(torch.load(brain_file, map_location=torch.device('cpu')))
    brain.eval()
    print(f"Successfully loaded '{brain_file}'")
except FileNotFoundError:
    print(f"ERROR: Could not find model file '{brain_file}'. Please upload it.")
    raise

# --- Set Up the Simulation Environment ---
p.connect(p.DIRECT)  # Use DIRECT mode for rendering without GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load the plane and robot
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF(
    "kuka_iiwa/model.urdf",
    [0, 0, 0],
    p.getQuaternionFromEuler([0, 0, 0])
)
END_EFFECTOR_LINK_INDEX = 6

# --- Initialize Frame Storage ---
frames = []

# --- Define the Target Position ---
target_pos = np.array([X,Y,Z])
print("Debug text location:", target_pos)

# --- Add Debug Text (Large "X") ---
p.addUserDebugText(
    text="X",
    textPosition=target_pos,
    textColorRGB=[0, 1, 0],  # Green
    textSize=8.0,            # Large size
    lifeTime=0               # Persistent
)

# --- Add Debug Lines (Forming a "+") ---
line_length = 0.2  # 20 cm
line_width = 10    # Thick lines
p.addUserDebugLine(
    lineFromXYZ=(target_pos - np.array([line_length/2, 0, 0])),
    lineToXYZ=(target_pos + np.array([line_length/2, 0, 0])),
    lineColorRGB=[0, 1, 0],
    lineWidth=line_width,
    lifeTime=0
)
p.addUserDebugLine(
    lineFromXYZ=(target_pos - np.array([0, line_length/2, 0])),
    lineToXYZ=(target_pos + np.array([0, line_length/2, 0])),
    lineColorRGB=[0, 1, 0],
    lineWidth=line_width,
    lifeTime=0
)

# --- Add a Green Sphere as the Target Marker ---
sphere_radius = 0.05  # 5 cm radius
sphere_visual_id = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=sphere_radius,
    rgbaColor=[0, 1, 0, 1]  # Green, fully opaque
)
sphere_id = p.createMultiBody(
    baseMass=0,                    # Static object
    baseVisualShapeIndex=sphere_visual_id,
    basePosition=target_pos
)

# --- Step Simulation Once to Initialize the Scene ---
p.stepSimulation()

# --- Set Up the Camera ---
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1.5, 1.5, 1.5],   # I moved the camera a bit closer too
    cameraTargetPosition=target_pos,   # Aim at the new target
    cameraUpVector=[0, 0, 1]
)
projection_matrix = p.computeProjectionMatrixFOV(
    fov=60.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=100.0
)


# --- Run the Simulation Loop ---
for i in range(240 * 3):  # 3 seconds at 240 fps
    # Get current joint states
    current_joint_states = [
        p.getJointState(robotId, j)[0]
        for j in range(p.getNumJoints(robotId))
    ]

    # Form the state vector
    state_vector = np.concatenate([current_joint_states, target_pos])
    state_tensor = torch.FloatTensor(state_vector)

    # Predict action using the policy network
    with torch.no_grad():
        action_tensor = brain(state_tensor)

    # Apply action to the robot's joints
    p.setJointMotorControlArray(
        bodyIndex=robotId,
        jointIndices=range(p.getNumJoints(robotId)),
        controlMode=p.POSITION_CONTROL,
        targetPositions=action_tensor.numpy()
    )

    # Capture the camera image
    img_data = p.getCameraImage(
        width=320,
        height=240,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )[2]
    frames.append(np.reshape(img_data, (240, 320, 4))[:, :, :3])  # RGB only

    # Step the simulation
    p.stepSimulation()

# --- Clean Up ---
p.disconnect()

# --- Save the Video ---
print("Simulation finished. Creating video…")
video_path = 'kuka_test_with_target.mp4'
imageio.mimsave(video_path, frames, fps=60)
print(f"Video saved to {video_path}")

# --- Display the Video ---
Video(video_path, embed=True, width=400)
