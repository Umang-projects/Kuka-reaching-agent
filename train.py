print("Starting training...")
start_time = time.time()

X = 0.650 * (math.cos(0))
Y = 0.65 * (math.sin(0))
Z = 0.4

for episode in range(80000):
    # reset
    p.resetBasePositionAndOrientation(robotId, [0,0,0], p.getQuaternionFromEuler([0,0,0]))
    target_pos = np.array([X,Y,Z])
    # Get the "Correct" Action using Inverse Kinematics
    ideal_joint_positions = p.calculateInverseKinematics(robotId, END_EFFECTOR_LINK_INDEX, target_pos)

    # Get the Current State
    current_joint_states = [p.getJointState(robotId, i)[0] for i in range(NUM_JOINTS)]
    state_vector = np.concatenate([current_joint_states, target_pos])

    # CHANGE 3: Move data tensors to the GPU
    state_tensor = torch.FloatTensor(state_vector).to(device)
    ideal_joint_positions_tensor = torch.FloatTensor(ideal_joint_positions).to(device)

    # Use Brain(NN) to Predict an Action
    predicted_joint_positions_tensor = brain(state_tensor)

    # Calculate Loss and Train
    optimizer.zero_grad()
    loss = loss_function(predicted_joint_positions_tensor, ideal_joint_positions_tensor)
    loss.backward()
    optimizer.step()

    # Apply the predicted action to the robot (no need to visualize here)
    p.setJointMotorControlArray(
        bodyIndex=robotId,
        jointIndices=range(NUM_JOINTS),
        controlMode=p.POSITION_CONTROL,
        targetPositions=predicted_joint_positions_tensor.cpu().detach().numpy() # Move back to CPU for numpy
    )
    p.stepSimulation()

    if episode % 5000 == 0:
        print(f"Episode: {episode}, Loss: {loss.item():.6f}")

p.disconnect()

end_time = time.time()
print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")

# --- 6. SAVE THE TRAINED MODEL ---
# The trained model will be saved in the Colab file system
torch.save(brain.state_dict(), "kuka_brain_gpu.pth")
