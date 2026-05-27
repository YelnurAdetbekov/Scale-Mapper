from roboflow import Roboflow
rf = Roboflow(api_key="oH28vEfk9W7kb3Nvx1VQ")
project = rf.workspace("yelnurs-workspace-uigac").project("scalemapper")
version = project.version(1)
dataset = version.download("yolov8", location="data/dataset")