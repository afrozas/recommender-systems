import numpy as np
from .svd import SVD

class CUR:
	""" Class to implement CUR Decomposition
	""" 
	def __init__(self, matrix):
		self.matrix = np.array(matrix)
		self.numRows = np.shape(self.matrix)[0]
		self.numCols = np.shape(self.matrix)[1]
		self.getProbDistribution()
		self.findR(1,2)
		self.findC(1,2)
		self.findU()
		self.multiply()

	def getProbDistribution(self):
		""" Gets the probability distribution for the rows and columns to be used in 
			random list generation.
		"""
		self.totalSum = sum(sum(self.matrix ** 2))
		self.rowSum = (sum((self.matrix.T)**2))/self.totalSum
		self.colSum = (sum(self.matrix ** 2))/self.totalSum

	def generateRandomNos(self, probDist, size, sampleSize, choice):
		""" The method generates sampleSsize number of random numbers which are in the range of 
			size. It samples out them using the given probability distribution
		"""
		if choice == 0: return np.random.choice(np.arange(0,size), sampleSize, p=probDist)
		else: return np.random.choice(np.arange(0,size), sampleSize, replace=False,p=probDist)

	def findC(self, choice, sampleSize):
		""" Method to compute C matrix in the CUR decomposition 
		"""
		rand_no = self.generateRandomNos(self.colSum, self.numCols, 2, choice)
		self.c_indices = rand_no
		self.C = (self.matrix[:,rand_no]).astype(float)
		colIdx, idx = 0, 0
		while colIdx < sampleSize:
			for rowIdx in range(0, self.numRows):
				self.C[rowIdx, colIdx] /= (sampleSize*self.colSum[rand_no[idx]])**0.5
			idx += 1
			colIdx += 1

	def findR(self, choice, sampleSize):
		""" Method to compute R matrix in the CUR decomposition 
		"""
		rand_no = self.generateRandomNos(self.rowSum, self.numRows, 2, choice)
		self.R = (self.matrix[rand_no,:]).astype(float)
		rowIdx, idx = 0, 0
		while rowIdx < sampleSize:
			for colIdx in range(0, self.numCols):
				self.R[rowIdx, colIdx] /= (sampleSize*self.rowSum[rand_no[idx]])**0.5
			idx += 1
			rowIdx += 1

	def findU(self):
		""" Method to compute U matrix in the CUR decomposition 
		"""
		self.U = self.R[:,self.c_indices]
		svd = SVD(self.U)
		Y = svd.V.T
		sigma_sq_plus = 1/(svd.sigma ** 2)
		sigma_sq_plus[sigma_sq_plus == np.inf] = 0
		sigma_sq_plus[sigma_sq_plus == -np.inf] = 0
		sigma_sq_plus[sigma_sq_plus == np.nan] = 0
		X = svd.U.T
		self.U = np.dot(Y,np.dot(sigma_sq_plus,X))
		self.U[self.U == np.inf] = 0
		self.U[self.U == -np.inf] = 0
		self.U[self.U == np.nan] = 0

	def multiply(self):
		""" Multiplies the U, V and sigma matrices to get the matrix of predicted ratings.
		"""
		self.result = np.dot((np.dot(self.C, self.U)), self.R)
		self.result[self.result == np.inf] = 0
		self.result[self.result == -np.inf] = 0
		self.result[self.result == np.nan] = 0
