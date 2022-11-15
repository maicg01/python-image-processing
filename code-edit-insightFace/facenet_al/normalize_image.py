
# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot as plt
import torch 
from torchvision import transforms
import os
from inceptionResnetV1 import InceptionResnetV1
from sklearn.preprocessing import Normalizer
 
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	
	image = Image.fromarray(pixels)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
 
# # load the photo and extract the face
# pixels = extract_face('/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/face.png')
# print(pixels.shape)

# specify folder to plot
folder = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/faceMTCNN/'
i = 1

def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in os.listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in os.listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# X, y = load_dataset(folder)
# print(X, y)

#crate faceEmbeddings


def get_normalized(face_array):
    # scale pixel values
    face_pixels = face_array.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    # std_adj = std.clamp(min=1.0/(float(face_pixels.numel())**0.5))
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    # samples = expand_dims(face_pixels, axis=0)

    return face_pixels

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

filname1 = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/faceMTCNN/person1/face.png'
filname2 = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/faceMTCNN/person1/face2.png'
filname3 = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/faceMTCNN/person2/face3.png'
filname4 = '/home/maicg/Documents/python-image-processing/code-edit-insightFace/facenet_al/faceMTCNN/person2/face4.png'

def computeCosin(filename1, filename2):
    img1 = extract_face(filename=filename1)
    img2 = extract_face(filename=filename2)
    nor_img1 = get_normalized(img1)
    nor_img2 = get_normalized(img2)
    print(nor_img2)

    convert_tensor = transforms.ToTensor()
    conv_img1=convert_tensor(nor_img1)
    conv_img2=convert_tensor(nor_img2)
    print(conv_img2.shape)

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img_embedding1 = resnet(conv_img1.unsqueeze(0))
    img_embedding2 = resnet(conv_img2.unsqueeze(0))

    
    print(img_embedding1)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(img_embedding1, img_embedding2)
    print("goc ti le giua anh 1 va 2: ", output)
    return output


output = computeCosin(filname1, filname2)
output = computeCosin(filname1, filname3)
output = computeCosin(filname1, filname4)
