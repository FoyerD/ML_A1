import kagglehub

# Download latest version
path = kagglehub.dataset_download("ulrikthygepedersen/speed-dating")

print("Path to dataset files:", path)