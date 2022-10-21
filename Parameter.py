import math
import torch
import torch.nn as nn

class GaussianParameter(nn.Module):
	def __init__(self, size):
		super().__init__()
		if isinstance(size, int):
			self.size = (size,)
		else:
			self.size = size

		self.mu = nn.Parameter(torch.zeros(self.size))
		## implement non-negative representation of sigma
		self.rho = nn.Parameter(torch.zeros(self.size))
		#self.sigma = nn.Parameter(torch.zeros(self.size))

		## init threshold as None to represent no pruning during testing or prediction
		self.threshold = None
		self.lockEpsilon = False

	## going to rely on layer module to init mu and sigma

	def forward(self):
		if not self.lockEpsilon:
			self.genEpsilon()
		## implement non-negative representation of sigma
		#self.value = self.mu  + self.sigma * self.epsilon
		self.value = self.mu  + self.getSigma() * self.epsilon
		## when model self.threshold is positive, make sure to set threshold at zero when training the network.
		if self.threshold is not None:
			## use strict inequality for dropping
			self.value[self.getLogitProb() < self.threshold] = 0 
		return self.value

	def setMixturePrior(self, pi, tau1, tau0):
		self.pi = pi # slab prob
		self.tau1 = tau1 # slab sd
		self.tau0 = tau0 # spike sd

	## interface for setting pruning threshold on inclusionProb
	def setThreshold(self, threshold):
		self.threshold = threshold

	def getMu(self):
		return self.mu

	def getRho(self):
		return self.rho

	def getSigma(self):
		## implement non-negative representation of sigma
		#return self.sigma
		return torch.log(1 + torch.exp(self.rho))

	def rollEpsilon(self):
		device = self.mu.get_device()
		if device == -1:
			device = 'cpu'
		self.epsilon = torch.randn(size = self.size, device =  device)

	def getEpsilon(self):
		return self.epsilon

	def genEpsilon(self):
		device = self.mu.get_device()
		if device == -1:
			device = 'cpu'
		if self.training:
			self.epsilon = torch.randn(size = self.size, device =  device)
		else:
			self.epsilon = torch.zeros(size = self.size, device =  device)

	def getA(self):
		return (self.mu**2 + (self.getSigma())**2)/(2*(self.tau1**2)) + math.log(self.tau1/self.pi)

	def getB(self):
		return (self.mu**2 + (self.getSigma())**2)/(2*(self.tau0**2)) + math.log(self.tau0/(1-self.pi))

	def getInclusionProb(self):
		return 1/(1 + torch.exp(self.getA()-self.getB()))

	def getLogitProb(self):
		return (self.getB() - self.getA())

	def getMixturePenalty(self):
		p = self.getInclusionProb()
		p = p.clamp(0.1**5, 1 - 0.1**5)
		return (p*(self.getA() + torch.log(p)) + (1-p)*(self.getB() + torch.log(1-p)) - torch.log(self.getSigma()))

	def getBlundellPenalty(self):
		return (-torch.log(self.getSigma()) - torch.log((self.pi/self.tau1)*torch.exp(-(self.value**2)/(2*self.tau1**2)) + ((1-self.pi)/self.tau0)*torch.exp(-(self.value**2)/(2*self.tau0**2))))

