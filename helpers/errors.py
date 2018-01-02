class Error:
	""" Class that provides error measures: RMS Error, TopK Precision and Spearman Rank Correlation
	"""
	def __init__(self, predictedRatings, trueRatings):
		self.predictedRatings = predictedRatings
		self.trueRatings = trueRatings

	def findRMSError(self, predictedRatings, trueRatings):
		""" Method to compute RMS Error given the actual and predicted ratings
		"""
		size = len(trueRatings)
		num = 0
		for i in range(size):
			num += abs(predictedRatings[i] - int(trueRatings[i])) ** 2
		num = (num**0.5)/size
		return num

	def topKPrecision(self, K=100):
		""" Finds the RMS Errror for the top K ratings among the data given 
		"""
		index = sorted(range(len(self.trueRatings)), key=lambda k: self.trueRatings[k], reverse=True)[:K]
		predictedRatings = []
		trueRatings = []
		for i in range(len(index)):
			trueRatings.append(self.trueRatings[i])
			predictedRatings.append(self.predictedRatings[i])
		return self.findRMSError(predictedRatings, trueRatings)

	def spearmanError(self):
		""" Calculates spearman rank correlation for the provided data
		"""
		size = len(self.trueRatings)
		num = 0
		for i in range(size):
			num += abs(self.predictedRatings[i] - int(self.trueRatings[i])) ** 2
		num = 6*num
		denum = size*(size*size-1)
		return (1 - (num/denum))
