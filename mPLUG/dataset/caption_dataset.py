# how to get input resolution
#
import json
import numpy as np
import time
import logging
import os
import random
import clip
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

def decode_int32(ann):
    ann = str(ann)
    server = str(int(ann[-1]) + 1)
    id_ = "0"*(9-len(ann[:-1]))+ann[:-1]
    assert len(id_) == 9
    ann = server+"/"+id_
    return ann

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    
class sherlock_dataset(Dataset):
    def __init__(self, data, args, training=False):
        self.args = args
        self.data = data
        self.id2data = {d['instance_id']: d for d in self.data}
        self.training = training
        if self.args.widescreen_processing in [0, 1]:
            self.preprocess = self._transform_train(args.image_res) if self.training else self._transform_test(args.image_res)
        else:
            self.preprocess = self._transform_train_pad(args.image_res) if self.training else self._transform_test_pad(args.image_res)

    def url2filepath(self, url):
        if 'VG_' in url:
            return self.args.vg_dir + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in self.args.vcr_dir:
                return self.args.vcr_dir + '/'.join(url.split('/')[-2:])
            else:
                return self.args.vcr_dir + '/'.join(url.split('/')[-3:])

    def hide_region(self, image, bboxes):
        image = image.convert('RGBA')
         #highlight mode
        overlay = Image.new('RGBA', image.size, '#00000000')
        draw = ImageDraw.Draw(overlay, 'RGBA')
       
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
           # highlight mode
            draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                            fill='#ff05cd3c', outline='#05ff37ff', width=3)
           
        image = Image.alpha_composite(image, overlay)

        return image

    def _transform_train(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomCrop(n_px),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_train_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def image_to_torch_tensor(self, image):
        if self.args.widescreen_processing == 1:
            width, height = image.size
            if width >= height:
                im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
                im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
            else:
                im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
                im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
            regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]
            image = torch.stack([self.preprocess(r) for r in regions], 0)
        else:
            image = self.preprocess(image)
        return image

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(self.url2filepath(c_data['inputs']['image']['url']))

        if self.args.hide_true_bbox > 0:
            image = self.hide_region(image, c_data['inputs']['bboxes'])

        # clue = clip.tokenize(c_data['inputs']['clue'], truncate=True).squeeze()
        clue = c_data['inputs']['clue']
        
       
        # caption = clip.tokenize('{}'.format(c_data['inputs']['clue']),
        #                             truncate=True).squeeze()
        caption = clue
        cid = c_data['instance_id']
        image = self.image_to_torch_tensor(image)
        return {'image':image, 'caption':caption, 'clue': clue, 'id': cid, "object_label": ""} #object_label is added just to be compatible with coco implementation

    def get(self, cid):
        return self.id2data[cid]

    def __len__(self):
        return len(self.data)


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
class nocaps_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.add_object = add_object
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        """ 
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
        """
        image_id = ann['img_id'] 
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    # file_str = bucket.get_object(file_path)
                    # file_buf = io.BytesIO()
                    # file_buf.write(file_str.read())
                    # file_buf.seek(0)
                    # file_buf = BytesIO(bucket.get_object(file_path).read())
                    # img_info = np.load(file_buf)
                    # file_buf.close()
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, image_id

class coco_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["filename"]
            sentences = each["sentences"]
            filepath = each["filepath"]
            if filepath == "val2014":
                file_root = "val2014_img"
            elif filepath == "train2014":
                file_root = "train2014_img"
            else:
                file_root = filepath
            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for sent in sentences:
                caption = sent["raw"]
                gold_caption.append(caption.lower())
            if self.add_object:
                object_list = each["object_label"].split("&&")
                new_object_list = list(set(object_list))
                new_object_list.sort(key=object_list.index)
                object_label = " ".join(new_object_list) 
            else:
                object_label = ""
            if is_train:
                for sent in sentences:
                    caption = sent["raw"].lower()
                    self.ann_new.append({"image": image_path, "caption": caption, "gold_caption": gold_caption, "object_label": object_label})
            else:
                self.ann_new.append({"image": image_path, "caption": sentences[0]["raw"].lower(), "gold_caption": gold_caption, "object_label": object_label})
        self.ann = self.ann_new
        del self.ann_new
            
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1] 
        object_label = ann['object_label']
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, caption, object_label, image_id, ann["gold_caption"]
class pretrain_dataset_4m(Dataset):
    def __init__(self, ann_file, transform, max_words=30, read_local_data=True, image_root="", epoch=None):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            bucket_name = "xxxxx"
            auth = oss2.Auth("xxxxx", "xxxxxx")
            self.bucket = oss2.Bucket(auth, "xxxxx", bucket_name)
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    time.sleep(0.1)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, caption
