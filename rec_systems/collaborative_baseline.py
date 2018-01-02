class CollaborativeBaselineFilter:
	""" Class that implements collaborative filtering with baseline approach.
	"""
	def __init__(self, ratings, cb):
		self.ratings = ratings
		self.MAX_ITEMS = 2000
		self.MAX_USERS = 1000
		self.filter = cb
		self.getBaselineEstimates()

	def getBaselineEstimates(self):
		""" Computes the baseline estimate 
		"""
		total_ratings = sum_of_ratings = 0

		self.user_mean = [0]*self.MAX_USERS
		self.item_mean = [0]*self.MAX_ITEMS
		self.user_count = [0]*self.MAX_USERS
		self.item_count = [0]*self.MAX_ITEMS

		for item in self.ratings:
			for user in self.ratings[item]:
				rating = int(self.ratings[item][user])
				self.user_mean[int(user)] += rating
				self.item_mean[int(item)] += rating
				self.user_count[int(user)] += 1
				self.item_count[int(item)] += 1

				total_ratings += 1
				sum_of_ratings += rating

		self.overall_mean = sum_of_ratings/total_ratings
		
		for idx in range(self.MAX_USERS):
			if self.user_count[idx] > 0:
				self.user_mean[idx] /= self.user_count[idx]
			else:
				self.user_mean[idx] = 0

		for idx in range(self.MAX_ITEMS):
			if self.item_count[idx] > 0:
				self.item_mean[idx] /= self.item_count[idx]
			else:
				self.item_mean[idx] = 0

	def predict(self, userID, itemID, N=100):
		""" Predicts the desired rating for some user and item.

			userID: ID of the user for whom rating is being predicted
			itemID: ID of the item for which rating is to be predicted
			N: number of similar items to be considered
		"""
		self.normalized_ratings = self.filter.normalized_ratings

		if itemID in self.ratings and userID in self.ratings[itemID]:
			print("Error: Rating already present")
			return -1

		if itemID not in self.ratings:
			return -1

		similarity = self.filter.getSimilarity(userID, itemID, N)

		similar_items = sorted(range(len(similarity)), key=lambda k: similarity[k], reverse=True)[1:N+1]
		numerator = denominator = 0

		for item in similar_items:
			if similarity[item] > 0 and str(item) in self.ratings and userID in self.ratings[str(item)]:
				baseline_estimate = self.item_mean[int(item)] + self.user_mean[int(userID)] - self.overall_mean
				numerator += (int(self.ratings[str(item)][userID]) - baseline_estimate)*similarity[item]
				denominator += similarity[item]

		rating = self.item_mean[int(itemID)] + self.user_mean[int(userID)] - self.overall_mean
		if denominator > 0:
			rating += numerator/denominator
		return rating 
