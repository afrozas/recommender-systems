class DatasetLoader:
	"""
	"""
	def __init__(self):
		"""
		"""
		self.ratings = {}

	def load_dataset(self):
		"""
		"""
		with open('data/dataset/ratings.dat') as dataset:
			# skipping first four lines from the text file as comments
			data = dataset.readlines()

		processed_data = []

		for item in data:
			processed_row = list(filter(None, item.split(':')))
			processed_data.append(processed_row[0:3])
		
		return processed_data


	def generate_dicts(self):
		"""
		"""
		processed_data = self.load_dataset()
		for row in processed_data:
			user, movie, rating = row[0:3]
			if user in self.ratings:
				self.ratings[user][movie] = rating
			else:
				self.ratings[user] = {movie: rating}
