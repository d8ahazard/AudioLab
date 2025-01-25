import os
dir_name = os.path.dirname(__file__)
joined = os.path.join(dir_name, "..")
app_path = os.path.abspath(joined)
output_path = os.path.join(app_path, "outputs")
model_path = os.path.join(app_path, "models")
