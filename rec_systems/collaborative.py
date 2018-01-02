class CollaborativeFilter:
	""" The class provides an implementation of collaborative
		filtering (item-item filtering) and predicts missing 
		entries.
	"""
	def __init__(self, ratings):
		self.ratings = ratings
		self.MAX_ITEMS = 2000
		self.MAX_USERS = 1000
		self.normalized_ratings = {}
		self.normalizeRatings()

	def normalizeRatings(self):
		""" The ratings mapping is normalized by subtracting from
			an entry the mean rating of the corresponding item.
		"""
		self.count = [0]*self.MAX_ITEMS
		self.summation = [0]*self.MAX_ITEMS
		self.sq_summation = [0]*self.MAX_ITEMS
		self.average_rating = [0]*self.MAX_ITEMS

		for item in self.ratings:
			for user in self.ratings[item]:
				self.summation[int(item)] += int(self.ratings[item][user])
				self.sq_summation[int(item)] += (int(self.ratings[item][user]))**2
				self.count[int(item)] += 1

		for item in self.ratings:
			self.average_rating[int(item)] = self.summation[int(item)]/self.count[int(item)]
			for user in self.ratings[item]:
				if item in self.normalized_ratings:
					self.normalized_ratings[item][user] = int(self.ratings[item][user]) - self.average_rating[int(item)]
				else:
					self.normalized_ratings[item] = {user: (int(self.ratings[item][user]) - self.average_rating[int(item)])}

	def getSimilarity(self, userID, itemID, N=10):
		""" gets the similarity of all items w.r.t to itemID 
		"""
		similarity = [0]*self.MAX_ITEMS
		denum = [0]*self.MAX_ITEMS
		users = self.normalized_ratings[itemID].keys()

		for user in users:
			for item in self.normalized_ratings:
				if user in self.normalized_ratings[item]:
					similarity[int(item)] += int(self.normalized_ratings[item][user]) * int(self.normalized_ratings[itemID][user])
					denum[int(item)] += int(self.normalized_ratings[item][user]) ** 2

		for item in self.normalized_ratings:
			try:
				similarity[int(item)] /= (denum[int(itemID)]**0.5)*(denum[int(item)]**0.5)
			except:
				pass
		return similarity

	def predict(self, userID, itemID, N=100):
		""" Predicts the desired rating for some user and item.

			userID: ID of the user for whom rating is being predicted
			itemID: ID of the item for which rating is to be predicted
			N: number of similar items to be considered
		"""
		if itemID in self.ratings and userID in self.ratings[itemID]:
			print("Error: Rating already present")
			return -1

		if itemID not in self.normalized_ratings:
			return self.average_rating[int(itemID)]

		similarity = self.getSimilarity(userID, itemID, N)

		similar_items = sorted(range(len(similarity)), key=lambda k: similarity[k], reverse=True)[1:N+1]
		numerator = denominator = 0

		for item in similar_items:
			if similarity[item] > 0 and str(item) in self.ratings and userID in self.ratings[str(item)]:
				numerator += int(self.ratings[str(item)][userID])*similarity[int(item)]
				denominator += similarity[int(item)]

		if denominator > 0:
			rating = numerator/denominator
		else:
			rating = self.average_rating[int(itemID)]
		return rating 
