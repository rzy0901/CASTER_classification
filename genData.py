import os
import shutil
import random

random.seed(0)

# "pic1": "push_pull_resize",
# "pic2": "beckoned_resize",
# "pic3": "rub_finger_resize",
# "pic4": "plug_resize",
# "pic5": "scale_resize",

def get_jpg_files(folder_path):
    jpg_files = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_files.add(file)
    return jpg_files

for model in ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']:
    if os.path.exists("./{}".format(model)):
        shutil.rmtree("./{}".format(model))
pics = ['pic1', 'pic2', 'pic3', 'pic4', 'pic5']

real_data = {f"pic{i}": get_jpg_files(f"./real_data/pic{i}") for i in range(1, 6)}
simulated_data = {f"pic{i}": get_jpg_files(f"./simulated_data/pic{i}") for i in range(1, 6)}
model1_data = {
    "test": {f"pic{j}": set() for j in range(1, 6)},
    "train": {f"pic{j}": set() for j in range(1, 6)}
}
model2_data = {
    "test": {f"pic{j}": set() for j in range(1, 6)},
    "train": {f"pic{j}": set() for j in range(1, 6)}
}
model3_data = {
    "test": {f"pic{j}": set() for j in range(1, 6)},
    "train": {f"pic{j}": set() for j in range(1, 6)}
}
model4_data = {
    "test": {f"pic{j}": set() for j in range(1, 6)},
    "train": {f"pic{j}": set() for j in range(1, 6)}
}
model5_data = {
    "test": {f"pic{j}": set() for j in range(1, 6)},
    "train": {f"pic{j}": set() for j in range(1, 6)}
}
model6_data = {
    "test": {f"pic{j}": set() for j in range(1, 6)},
    "train": {f"pic{j}": set() for j in range(1, 6)}
}

## train: real_data, test: real_data
## model5_data: 60 train images for each class in real_data, 40 test images for each class in real_data
for pic in pics:
    random_sample_real_60 = set(random.sample(real_data[pic], 60))
    random_sample_real_40 = real_data[pic] - random_sample_real_60
    model5_data["test"][pic] = random_sample_real_40
    model5_data["train"][pic] = random_sample_real_60

## train: simulated_data, test: simulated_data
## model6_data: 60 train images for each class in simulated_data, 40 test images for each class in simulated_data
for pic in pics:
    random_sample_sim_60 = set(random.sample(simulated_data[pic], 60))
    random_sample_sim_40 = simulated_data[pic] - random_sample_sim_60
    model6_data["test"][pic] = random_sample_sim_40
    model6_data["train"][pic] = random_sample_sim_60

real_data_not_in_model5_test = {
    "pic1":real_data["pic1"] - model5_data["test"]["pic1"],
    "pic2":real_data["pic2"] - model5_data["test"]["pic2"],
    "pic3":real_data["pic3"] - model5_data["test"]["pic3"],
    "pic4":real_data["pic4"] - model5_data["test"]["pic4"],
    "pic5":real_data["pic5"] - model5_data["test"]["pic5"],
}

# model1_data: 60 train images for each class in simulated class, 40 test images for each class in real class (same test images as model5_data)
for pic in pics:
    random_sample_sim_60 = set(random.sample(simulated_data[pic], 60))
    model1_data["test"][pic] = model5_data["test"][pic].copy()
    model1_data["train"][pic] = random_sample_sim_60

## model2_data: 50 train images for each class in simulated class, 10 train images for each class in real class, 40 test images for each class in real class (same test images as model5_data)
for pic in pics:
    random_sample_sim_50 = set(random.sample(simulated_data[pic], 50))
    random_sample_real_10 = set(random.sample(real_data_not_in_model5_test[pic], 10))
    model2_data["test"][pic] = model5_data["test"][pic].copy()
    model2_data["train"][pic] = random_sample_sim_50.union(random_sample_real_10)

## model3_data: 40 train images for each class in simulated class, 20 train images for each class in real class, 40 test images for each class in real class (same test images as model5_data)
for pic in pics:
    random_sample_sim_40 = set(random.sample(simulated_data[pic], 40))
    random_sample_real_20 = set(random.sample(real_data_not_in_model5_test[pic], 20))
    model3_data["test"][pic] = model5_data["test"][pic].copy()
    model3_data["train"][pic] = random_sample_sim_40.union(random_sample_real_20)

## model4_data: 30 train images for each class in simulated class, 30 train images for each class in real class, 40 test images for each class in real class (same test images as model5_data)
for pic in pics:
    random_sample_sim_30 = set(random.sample(simulated_data[pic], 30))
    random_sample_real_30 = set(random.sample(real_data_not_in_model5_test[pic], 30))
    model4_data["test"][pic] = model5_data["test"][pic].copy()
    model4_data["train"][pic] = random_sample_sim_30.union(random_sample_real_30)

## Verification
model_datas = [model1_data, model2_data, model3_data, model4_data, model5_data, model6_data]
for i,model_data in enumerate(model_datas):
    for pic in pics:
        overlap_files = model_data["test"][pic].intersection(model_data["train"][pic])
        print("model{}中{}的测试及训练重叠的文件名集合:".format(i+1, pic), overlap_files)
        
## Save the data
for i, model_data in enumerate(model_datas):
    os.mkdir("./model{}".format(i+1))
    os.mkdir("./model{}/test".format(i+1))
    os.mkdir("./model{}/train".format(i+1))
    for pic in pics:
        os.mkdir("./model{}/test/{}".format(i+1,pic))
        os.mkdir("./model{}/train/{}".format(i+1,pic))
        for filename in model_data["test"][pic]:
            src_path = './real_data' if filename.endswith('_r.jpg') else './simulated_data'
            src_path = os.path.join(src_path, pic, filename)
            dst_path = os.path.join("./model{}/test".format(i+1), pic, filename)
            # print("src_path:", src_path)
            # print("dst_path:", dst_path)
            shutil.copyfile(src_path, dst_path)
        for filename in model_data["train"][pic]:
            src_path = './real_data' if filename.endswith('_r.jpg') else './simulated_data'
            src_path = os.path.join(src_path, pic, filename)
            dst_path = os.path.join("./model{}/train".format(i+1), pic, filename)
            # print("src_path:", src_path)
            # print("dst_path:", dst_path)
            shutil.copyfile(src_path, dst_path)
                