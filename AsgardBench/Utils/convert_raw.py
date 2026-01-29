import json
import os

from AsgardBench import constants as c
from AsgardBench.plan import RawPlan

# If for some reason the raw_plans need to be re-converted because of a format change,
# this will go through all raw_plan.json files in the best_of directory and convert
# them to plan.json files, so you don't have to re-run the entire generation process.
new_plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"
if not os.path.exists(new_plans_dir):
    print("No plans found.")


all_directories = os.listdir(new_plans_dir)
for directory in all_directories:
    # Read in raw_plan.json file
    raw_plan_path = os.path.join(new_plans_dir, directory, "raw_plan.json")
    if not os.path.exists(raw_plan_path):
        print(f"Skipping {directory} - raw_plan.json not found.")
        continue

    # Read and convert to Plan object
    with open(raw_plan_path, "r", encoding="utf-8") as f:
        plan_json = json.load(f)
        plan = RawPlan.from_dict(plan_json)

        action_plan = plan.plan_from_raw_plan()

    # Write to plan.json
    plan_path = os.path.join(new_plans_dir, directory, "plan2.json")
    with open(plan_path, "w", encoding="utf-8") as s:
        action_json = action_plan.to_dict()
        json.dump(action_json, s)
        s.flush()
        print(f"Converted {directory}")
