import cadquery as cq

# Parameters for adjustability
wristband_width = 30
marker_holder_diameter = 15
strap_hole_diameter = 6

# Create the hand back base with adjustable slots
hand_back = (
    cq.Workplane("XY")
    .rect(100, 80)  # Base size for the hand back
    .extrude(3)  # Thickness
    .faces(">Z")
    .workplane()
    .center(0, -30)
    .hole(strap_hole_diameter)  # Slot for wristband
    .center(0, 60)
    .hole(strap_hole_diameter)  # Slot for fingers
)

# Add marker holders with detachable mounts
marker_positions = [(-30, 30), (0, 40), (30, 30)]
for pos in marker_positions:
    marker_holder = (
        cq.Workplane("XY")
        .center(pos[0], pos[1])
        .circle(marker_holder_diameter / 2)
        .extrude(10)  # Height for the holder
        .faces(">Z")
        .hole(5)  # Inner hole for marker attachment
    )
    hand_back = hand_back.union(marker_holder)

# Add holes for adjustable straps
adjustable_straps = [(-40, 0), (40, 0)]
for pos in adjustable_straps:
    hand_back = (
        hand_back.faces("<Z")
        .workplane()
        .center(pos[0], pos[1])
        .rect(10, 3)  # Slots for adjustable straps
        .cutBlind(-2)
    )

# Export the model
cq.exporters.export(hand_back, "adjustable_vicon_hand_holder.step")
