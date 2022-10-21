import math
import warnings

import torch
from torch import Tensor
#from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F
from torch.nn import init
#from .lazy import LazyModuleMixin
from torch.nn import Module
from ..utils import _single, _pair, _triple, _reverse_repeat_tuple
#from torch._torch_docs import reproducibility_notes

from ..common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple



from MVB.nn.modules.Parameter import GaussianParameter


class _GaussianConvNd(Module):

	__constants__ = ['stride', 'padding', 'dilation', 'groups',
					 'padding_mode', 'output_padding', 'in_channels',
					 'out_channels', 'kernel_size']
	__annotations__ = {'bias': Optional[torch.Tensor]}

	_in_channels: int
	out_channels: int
	kernel_size: Tuple[int, ...]
	stride: Tuple[int, ...]
	padding: Tuple[int, ...]
	dilation: Tuple[int, ...]
	transposed: bool
	output_padding: Tuple[int, ...]
	groups: int
	padding_mode: str
	weight: Tensor
	bias: Optional[Tensor]


	def __init__(self,
				 in_channels: int,
				 out_channels: int,
				 kernel_size: _size_1_t,
				 stride: _size_1_t,
				 padding: _size_1_t,
				 dilation: _size_1_t,
				 transposed: bool,
				 output_padding: _size_1_t,
				 groups: int,
				 bias: bool,
				 padding_mode: str) -> None:
		super(_GaussianConvNd, self).__init__()
		if in_channels % groups != 0:
			raise ValueError('in_channels must be divisible by groups')
		if out_channels % groups != 0:
			raise ValueError('out_channels must be divisible by groups')
		valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
		if padding_mode not in valid_padding_modes:
			raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
				valid_padding_modes, padding_mode))
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.transposed = transposed
		self.output_padding = output_padding
		self.groups = groups
		self.padding_mode = padding_mode
		
		self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
		if transposed:
			self.weight = GaussianParameter((
				in_channels, out_channels // groups, *kernel_size))
		else:
			self.weight = GaussianParameter((
				out_channels, in_channels // groups, *kernel_size))

		self.bias = GaussianParameter(out_channels)

		self.reset_parameters()

	def reset_parameters(self) -> None:
		init.kaiming_uniform_(self.weight.getMu(), a=math.sqrt(5))
		## implement non-negative representation of sigma
		#init.constant_(self.weight.getSigma(), 0.01)
		init.constant_(self.weight.getRho(), -4)

		fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight.getMu())
		bound = 1 / math.sqrt(fan_in)
		init.uniform_(self.bias.getMu(), -bound, bound)
		## implement non-negative representation of sigma
		#init.constant_(self.bias.getSigma(), 0.01)
		init.constant_(self.bias.getRho(), -4)

	def extra_repr(self):
		s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
			 ', stride={stride}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.output_padding != (0,) * len(self.output_padding):
			s += ', output_padding={output_padding}'
		if self.groups != 1:
			s += ', groups={groups}'
		if self.bias is None:
			s += ', bias=False'
		if self.padding_mode != 'zeros':
			s += ', padding_mode={padding_mode}'
		return s.format(**self.__dict__)

	def __setstate__(self, state):
		super(_PolarConvNd, self).__setstate__(state)
		if not hasattr(self, 'padding_mode'):
			self.padding_mode = 'zeros'




class GaussianConv2d(_GaussianConvNd):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: _size_2_t,
		stride: _size_2_t = 1,
		padding: _size_2_t = 0,
		dilation: _size_2_t = 1,
		groups: int = 1,
		bias: bool = True,
		padding_mode: str = 'zeros'  # TODO: refine this type
	):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(GaussianConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), groups, bias, padding_mode)

	def _conv_forward(self, input: Tensor, weight, bias):
		if self.padding_mode != 'zeros':
			padInput = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
			return F.conv2d(input, weight(), bias(), self.stride,
						_pair(0), self.dilation, self.groups)

		return F.conv2d(input, weight(), bias(), self.stride,
						self.padding, self.dilation, self.groups)

	def forward(self, input: Tensor) -> Tensor:
		return self._conv_forward(input, self.weight, self.bias)