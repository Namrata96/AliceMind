import zipfile
from tqdm import tqdm

targetdir = "/scratch/nm3571/multimodal/data/sherlock/vcr1_final/"
with zipfile.ZipFile("/scratch/nm3571/multimodal/data/sherlock/vcr1images.zip","r") as zip_ref:
    for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
        zip_ref.extract(member=file, path=targetdir)


# targetdir2 = "/scratch/nm3571/multimodal/data/sherlock/images_final/"
# with zipfile.ZipFile("/scratch/nm3571/multimodal/data/sherlock/images.zip","r") as zip_ref:
#     for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
#         zip_ref.extract(member=file, path=targetdir2)

# with zipfile.ZipFile("/scratch/nm3571/multimodal/data/sherlock/images2.zip","r") as zip_ref:
#     for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
#         zip_ref.extract(member=file, path=targetdir2)