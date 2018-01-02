import numpy as np

class SVD:
	""" Class to implement Singular Value Decomposition
	"""
	def __init__(self, matrix):
		self.matrix = matrix
		self.MAX_USERS = len(self.matrix)
		self.MAX_ITEMS = len(self.matrix[0])
		self.findU()
		self.findV()
		self.findSigma()
		self.tryDimReduction()
		self.multiply()

	def findU(self):
		""" finds the U matrix for SVD decomposition
		"""
		A = np.array(self.matrix)
		AAt = np.dot(A,A.T)
		self.eigen_values_AAt, self.eigen_vectors_AAt = np.linalg.eigh(AAt)
		self.rank_AAt = np.linalg.matrix_rank(AAt)
		self.index_AAt = sorted(range(len(self.eigen_values_AAt)),
							 key=lambda k: self.eigen_values_AAt[k], reverse=True)[:self.rank_AAt]
		self.U = np.zeros(shape=(self.MAX_USERS,self.rank_AAt))
		self.eigen_values_AAt = self.eigen_values_AAt[::-1] 
		self.eigen_values_AAt = self.eigen_values_AAt[:self.rank_AAt]

		for i in range(self.rank_AAt):
			self.U[:,i] = self.eigen_vectors_AAt[:,self.index_AAt[i]]

	def findSigma(self):
		""" finds the sigma matrix for SVD decomposition
		"""
		self.sigma = np.zeros(shape=(self.rank_AAt, self.rank_AAt))
		for i in range(self.rank_AAt):
			self.sigma[i,i] = self.eigen_values_AAt[i] ** 0.5

	def findV(self):
		""" finds the V matrix for SVD decomposition
		"""
		A = np.array(self.matrix)
		AtA = (A.T).dot(A)
		self.eigen_values_AtA, self.eigen_vectors_AtA = np.linalg.eigh(AtA)
		self.rank_AtA = np.linalg.matrix_rank(AtA)
		self.index_AtA = sorted(range(len(self.eigen_values_AtA)), key=lambda k: self.eigen_values_AtA[k], reverse=True)[:self.rank_AtA]
		self.V = np.zeros(shape=(self.MAX_ITEMS,self.rank_AtA))
		self.eigen_values_AtA = self.eigen_values_AtA[::-1]
		self.eigen_values_AtA = self.eigen_values_AtA[:self.rank_AtA]

		for i in range(self.rank_AtA):
			self.V[:,i] = self.eigen_vectors_AtA[:,self.index_AtA[i]]
		self.V = self.V.T

	def tryDimReduction(self):
		""" Tries to reduce the dimension of the matrics by following 90 %
			retain energy rule.
		"""
		while True:
			total_E = 0
			size = np.shape(self.sigma)[0]
			for i in range(size):
				total_E += self.sigma[i,i]**2
			retained_E = 0
			if size > 0:
				retained_E = total_E - self.sigma[size-1,size-1]**2
			if total_E == 0 or retained_E/total_E < 0.9:
				break
			else:
				self.U = self.U[:,:-1:]
				self.V = self.V[:-1,:]
				self.sigma = self.sigma[:,:-1]
				self.sigma = self.sigma[:-1,:]

	def multiply(self):
		""" Multiplies the U, V and sigma matrices to get the matrix of predicted ratings.
		"""
		self.result = np.dot((np.dot(self.U, self.sigma)), self.V)

