import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from MVB.nn.modules.Parameter import GaussianParameter

class GaussianLinear(nn.Module):
	def __init__(self, size_in, size_out):
		super().__init__()
		self.size_in = size_in
		self.size_out = size_out

		self.weight = GaussianParameter((size_out, size_in))
		self.bias = GaussianParameter(size_out)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.weight.getMu(), a=math.sqrt(5))
		#nn.init.kaiming_uniform_(self.weight.getSigma(), a=math.sqrt(5))
		## implement non-negative representation of sigma
		#nn.init.constant_(self.weight.getSigma(), 0.01)
		nn.init.constant_(self.weight.getRho(), -4)

		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.getMu())
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.bias.getMu(), -bound, bound)
		#nn.init.uniform_(self.bias.getSigma(), -bound, bound)
		## implement non-negative representation of sigma
		#nn.init.constant_(self.bias.getSigma(), 0.01)
		nn.init.constant_(self.bias.getRho(), -4)


	def forward(self, input):
		return F.linear(input, self.weight(), self.bias())
