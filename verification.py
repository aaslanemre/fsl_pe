import json

# Path to your new JSON
json_path = 'data/mpii/annotations/mpii_train.json'

with open(json_path, 'r') as f:
    data = json.load(f)

# The MMPose format usually stores things in a list
# Let's check the first entry for an activity label
first_entry = data[0]
print(f"Image: {first_entry['img_paths']}")
print(f"Joints Shape: {len(first_entry['joint_self'])} points")

# Check if we have activity information
if 'extra_info' in first_entry or 'activity_id' in first_entry:
    print("Success: Activity metadata found!")
else:
    print("Note: Activity labels might be in a separate mapping file.")
