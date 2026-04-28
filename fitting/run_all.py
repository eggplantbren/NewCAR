import os
import json
import dnest4
import shutil

os.makedirs("results", exist_ok=True)

modes = [
    (False, "informative", "informative_free"),
    (True,  "informative", "informative_fixed"),
    (False, "flat",        "flat_free"),
    (True,  "flat",        "flat_fixed"),
]

def write_config(fixed_mean, prior):
    with open("config.json", "w") as f:
        json.dump({"fixed_mean": fixed_mean, "prior": prior}, f)

def run_cpp(tag, seed):
    os.system(f"./main -s {seed}")
    dnest4.postprocess(plot=False, rng_seed=123)
    shutil.move("posterior_sample.txt", f"results/posterior_sample_{tag}.txt")

seed = 1
for fixed_mean, prior, tag in modes:
    write_config(fixed_mean, prior)
    run_cpp(tag, seed)
    seed += 1
