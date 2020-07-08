import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

class SuperResolutionNN(nn.Module):
	def __init__(self, upscale_factor, inplace=False):
		super(SuperResolutionNN, self).__init__()

		self.relu  = nn.ReLU(inplace=inplace)
		self.conv1 = nn.Conv2d(1, 64, (5,5), (1,1), (2,2))
		self.conv2 = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))
		self.conv3 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
		self.conv4 = nn.Conv2d(32, upscale_factor**2, (3,3), (1,1), (1,1))
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

		self._init_weights()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.pixel_shuffle(self.conv4(x))
		return x

	def _init_weights(self):
		init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv4.weight)

# model
torch_model = SuperResolutionNN(upscale_factor=3)

model_url= "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"
batch_size = 1
#batch_size = 0

map_location = lambda storage, loc: storage
if torch.cuda.is_available():
	map_location = None

torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
torch_model.eval()

x = torch.randn(batch_size, 1,112,112, requires_grad=True)
#x = torch.randn(batch_size, 1,224,224, requires_grad=True)
torch_out = torch_model(x)

print("[-] torch model:\t{}".format(torch_model))
for i, layer in enumerate(torch_model.modules()):
	print("layer {}:\t{}".format(i, layer))
for i, name in enumerate(torch_model.state_dict()):
	print("state_dict {}:\t{}".format(i, name))
for i, (name,param) in enumerate(torch_model.named_parameters()):
	print("named_parameters {}:\t{}".format(i, name))

torch.onnx.export(torch_model, x, "sr.onnx", 
	export_params=True, opset_version=10, do_constant_folding=True, 
	input_names = ['hello'], output_names = ['world'], 
	#input_names = ['input'], output_names = ['output'], 
	#dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
	dynamic_axes = {'hello': {0: 'batch_size'}, 'world': {0: 'batch_size'}})


# usage
# python convert-super-resolution.py
